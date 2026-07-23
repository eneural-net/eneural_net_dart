import 'dart:js_interop';
import 'dart:typed_data';

import '../native_backend.dart';
import 'webgpu_accelerator.dart';
import 'webgpu_bindings.dart';
import 'webgpu_shaders.dart';

/// Requests a WebGPU device and builds a [WebGpuAccelerator], or returns `null`
/// when WebGPU is unavailable (no `navigator.gpu`, no adapter, or an error).
Future<WebGpuAccelerator?> resolveWebGpuAccelerator({
  required NativeNetworkSpec spec,
  required bool rprop,
}) async {
  final gpu = navigator.gpu;
  if (gpu == null) return null;
  try {
    final adapter = await gpu.requestAdapter().toDart;
    if (adapter == null) return null;
    final device = await adapter.requestDevice().toDart;
    return _WebGpuAcceleratorImpl(device, spec, rprop);
  } catch (_) {
    return null;
  }
}

/// A precreated compute dispatch: pipeline + bind group + its uniform buffer.
class _Pass {
  final GPUComputePipeline pipeline;
  final GPUBindGroup bindGroup;
  final GPUBuffer uniform;
  final int dispatchX;
  final int dispatchY;
  _Pass(
    this.pipeline,
    this.bindGroup,
    this.uniform,
    this.dispatchX,
    this.dispatchY,
  );
}

class _WebGpuAcceleratorImpl implements WebGpuAccelerator {
  final GPUDevice device;
  final NativeNetworkSpec spec;

  late final GPUQueue queue;

  int get numLayers => spec.numLayers;
  List<int> get sizes => spec.neuronCounts;

  // Weight-space buffers (per connection L -> L+1).
  final List<GPUBuffer> _w = [];
  final List<GPUBuffer> _g = [];
  final List<GPUBuffer> _gPrev = [];
  final List<GPUBuffer> _prevDelta = [];
  final List<GPUBuffer> _lastUpdate = [];
  final List<int> _wCounts = [];

  // Batched activation buffers (per layer, [numSamples x sizes[L]]).
  final List<GPUBuffer> _out = [];
  final List<GPUBuffer> _delta = [];

  // Single-sample inference buffers.
  final List<GPUBuffer> _outInf = [];
  late final GPUBuffer _infStaging;

  GPUBuffer? _inputs;
  GPUBuffer? _targets;
  GPUBuffer? _errBuf;
  GPUBuffer? _errStaging;
  GPUBuffer? _weightsStaging;

  int _numSamples = 0;
  int _inputSize = 0;
  int _outputSize = 0;

  bool _useRprop;
  double _rDelta0 = 0.10;
  double _rMin = 1.0e-6;
  double _rMax = 50.0;
  double _rEtaPlus = 1.2;
  double _rEtaMinus = 0.5;

  double _globalLearnError = 1.0;
  double _lastGlobalLearnError = 1.0;

  // Pipelines.
  late final GPUComputePipeline _pZero;
  late final GPUComputePipeline _pCopy;
  late final GPUComputePipeline _pLoad;
  late final GPUComputePipeline _pForward;
  late final GPUComputePipeline _pOutputDelta;
  late final GPUComputePipeline _pBackprop;
  late final GPUComputePipeline _pGradient;
  late final GPUComputePipeline _pUpdateBp;
  late final GPUComputePipeline _pUpdateRprop;

  // Precreated per-epoch passes.
  final List<_Pass> _pass1 = [];
  final List<_Pass> _updatePasses = [];
  final List<int> _updateWCounts = [];
  bool _passesBuilt = false;

  _WebGpuAcceleratorImpl(this.device, this.spec, this._useRprop) {
    queue = device.queue;

    _pZero = _pipeline(wgslZero);
    _pCopy = _pipeline(wgslCopy);
    _pLoad = _pipeline(wgslLoadInput);
    _pForward = _pipeline(wgslForward);
    _pOutputDelta = _pipeline(wgslOutputDelta);
    _pBackprop = _pipeline(wgslBackprop);
    _pGradient = _pipeline(wgslGradient);
    _pUpdateBp = _pipeline(wgslUpdateBp);
    _pUpdateRprop = _pipeline(wgslUpdateRprop);

    const stCopy =
        GPUBufferUsage.storage |
        GPUBufferUsage.copyDst |
        GPUBufferUsage.copySrc;
    const st = GPUBufferUsage.storage | GPUBufferUsage.copyDst;

    for (var l = 0; l < numLayers - 1; ++l) {
      final n = sizes[l] * sizes[l + 1];
      _wCounts.add(n);
      _w.add(_buffer(n, stCopy));
      _g.add(_buffer(n, st));
      _gPrev.add(_buffer(n, st));
      _prevDelta.add(_buffer(n, st, fill: 0.10));
      _lastUpdate.add(_buffer(n, st));
    }
    for (var l = 0; l < numLayers; ++l) {
      _outInf.add(_buffer(sizes[l], stCopy));
    }
    _infStaging = _buffer(
      sizes[numLayers - 1],
      GPUBufferUsage.mapRead | GPUBufferUsage.copyDst,
    );
  }

  GPUComputePipeline _pipeline(String code) {
    final module = device.createShaderModule(
      GPUShaderModuleDescriptor(code: code),
    );
    return device.createComputePipeline(
      GPUComputePipelineDescriptor(
        layout: 'auto'.toJS,
        compute: GPUProgrammableStage(module: module, entryPoint: 'main'),
      ),
    );
  }

  GPUBuffer _buffer(int floats, int usage, {double? fill}) {
    final b = device.createBuffer(
      GPUBufferDescriptor(size: floats * 4, usage: usage),
    );
    if (fill != null && fill != 0.0) {
      final data = Float32List(floats)..fillRange(0, floats, fill);
      queue.writeBuffer(b, 0, data.toJS);
    }
    return b;
  }

  JSUint8Array _params(List<int> ints, [List<double> floats = const []]) {
    final bd = ByteData(64);
    for (var i = 0; i < ints.length && i < 8; ++i) {
      bd.setInt32(i * 4, ints[i], Endian.little);
    }
    for (var i = 0; i < floats.length && i < 8; ++i) {
      bd.setFloat32(32 + i * 4, floats[i], Endian.little);
    }
    return bd.buffer.asUint8List().toJS;
  }

  _Pass _makePass(
    GPUComputePipeline pipe,
    List<GPUBuffer> storages,
    JSUint8Array params,
    int gridX,
    int gridY,
    int wgX,
    int wgY,
  ) {
    final uniform = device.createBuffer(
      GPUBufferDescriptor(
        size: 64,
        usage: GPUBufferUsage.uniform | GPUBufferUsage.copyDst,
      ),
    );
    queue.writeBuffer(uniform, 0, params);

    final entries = <GPUBindGroupEntry>[];
    for (var i = 0; i < storages.length; ++i) {
      entries.add(
        GPUBindGroupEntry(
          binding: i,
          resource: GPUBufferBinding(buffer: storages[i]),
        ),
      );
    }
    entries.add(
      GPUBindGroupEntry(
        binding: storages.length,
        resource: GPUBufferBinding(buffer: uniform),
      ),
    );

    final bindGroup = device.createBindGroup(
      GPUBindGroupDescriptor(
        layout: pipe.getBindGroupLayout(0),
        entries: entries.toJS,
      ),
    );

    return _Pass(
      pipe,
      bindGroup,
      uniform,
      (gridX + wgX - 1) ~/ wgX,
      (gridY + wgY - 1) ~/ wgY,
    );
  }

  @override
  int get weightsLength {
    var t = 0;
    for (final c in _wCounts) {
      t += c;
    }
    return t;
  }

  @override
  void configBackprop() {
    _useRprop = false;
  }

  @override
  void configRProp(
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  ) {
    _useRprop = true;
    _rDelta0 = delta0;
    _rMin = deltaMin;
    _rMax = deltaMax;
    _rEtaPlus = etaPlus;
    _rEtaMinus = etaMinus;
  }

  @override
  Future<void> setSamples(
    int numSamples,
    int inputSize,
    int outputSize,
    Float32List inputs,
    Float32List targets,
  ) async {
    _numSamples = numSamples;
    _inputSize = inputSize;
    _outputSize = outputSize;

    const st = GPUBufferUsage.storage | GPUBufferUsage.copyDst;
    _inputs = _buffer(numSamples * inputSize, st);
    _targets = _buffer(numSamples * outputSize, st);
    _errBuf = _buffer(
      numSamples * outputSize,
      GPUBufferUsage.storage | GPUBufferUsage.copySrc,
    );
    _errStaging = _buffer(
      numSamples * outputSize,
      GPUBufferUsage.mapRead | GPUBufferUsage.copyDst,
    );
    _weightsStaging = _buffer(
      weightsLength,
      GPUBufferUsage.mapRead | GPUBufferUsage.copyDst,
    );

    queue.writeBuffer(_inputs!, 0, inputs.toJS);
    queue.writeBuffer(_targets!, 0, targets.toJS);

    _out.clear();
    _delta.clear();
    for (var l = 0; l < numLayers; ++l) {
      _out.add(_buffer(numSamples * sizes[l], GPUBufferUsage.storage));
      _delta.add(_buffer(numSamples * sizes[l], GPUBufferUsage.storage));
    }

    _buildPasses();
  }

  void _buildPasses() {
    _pass1.clear();
    _updatePasses.clear();
    _updateWCounts.clear();
    final last = numLayers - 1;
    final m = _numSamples;
    final errCount = _numSamples * _outputSize;

    // 1. zero errBuf
    _pass1.add(
      _makePass(_pZero, [_errBuf!], _params([errCount]), errCount, 1, 64, 1),
    );

    // 2. copy g -> gPrev
    for (var l = 0; l < numLayers - 1; ++l) {
      _pass1.add(
        _makePass(
          _pCopy,
          [_gPrev[l], _g[l]],
          _params([_wCounts[l]]),
          _wCounts[l],
          1,
          64,
          1,
        ),
      );
    }

    // 3. load input
    final biasIndex0 = spec.withBias[0] ? sizes[0] - 1 : -1;
    _pass1.add(
      _makePass(
        _pLoad,
        [_out[0], _inputs!],
        _params([sizes[0], _inputSize, biasIndex0, m]),
        sizes[0],
        m,
        8,
        8,
      ),
    );

    // 4. forward
    for (var l = 0; l < numLayers - 1; ++l) {
      final nextIsOutput = (l + 1) == last;
      final biasIndex = (!nextIsOutput && spec.withBias[l + 1])
          ? sizes[l + 1] - 1
          : -1;
      _pass1.add(
        _makePass(
          _pForward,
          [_w[l], _out[l], _out[l + 1]],
          _params(
            [sizes[l], sizes[l + 1], spec.activationIds[l + 1], biasIndex, m],
            [spec.activationScale[l + 1]],
          ),
          sizes[l + 1],
          m,
          8,
          8,
        ),
      );
    }

    // 5. output delta
    _pass1.add(
      _makePass(
        _pOutputDelta,
        [_out[last], _targets!, _delta[last], _errBuf!],
        _params(
          [sizes[last], spec.activationIds[last], m],
          [spec.flatSpots[last]],
        ),
        sizes[last],
        m,
        8,
        8,
      ),
    );

    // 6. gradient + backprop (high -> low)
    for (var l = numLayers - 2; l >= 0; --l) {
      _pass1.add(
        _makePass(
          _pGradient,
          [_g[l], _out[l], _delta[l + 1]],
          _params([sizes[l], sizes[l + 1], m]),
          sizes[l + 1],
          sizes[l],
          8,
          8,
        ),
      );
      if (l > 0) {
        _pass1.add(
          _makePass(
            _pBackprop,
            [_w[l], _delta[l + 1], _out[l], _delta[l]],
            _params(
              [sizes[l], sizes[l + 1], spec.activationIds[l], 1, m],
              [spec.flatSpots[l]],
            ),
            sizes[l],
            m,
            8,
            8,
          ),
        );
      }
    }

    // Update passes (params rewritten each epoch).
    for (var l = 0; l < numLayers - 1; ++l) {
      if (_useRprop) {
        _updatePasses.add(
          _makePass(
            _pUpdateRprop,
            [_w[l], _g[l], _gPrev[l], _prevDelta[l], _lastUpdate[l]],
            _params([0, _wCounts[l]], [_rEtaPlus, _rEtaMinus, _rMin, _rMax]),
            _wCounts[l],
            1,
            64,
            1,
          ),
        );
      } else {
        _updatePasses.add(
          _makePass(
            _pUpdateBp,
            [_w[l], _g[l], _prevDelta[l]],
            _params([_wCounts[l]], [0.0, 0.0]),
            _wCounts[l],
            1,
            64,
            1,
          ),
        );
      }
      _updateWCounts.add(_wCounts[l]);
    }

    _passesBuilt = true;
  }

  @override
  Future<void> setWeights(Float32List weights) async {
    var offset = 0;
    for (var l = numLayers - 2; l >= 0; --l) {
      final count = _wCounts[l];
      final segment = Float32List.sublistView(weights, offset, offset + count);
      queue.writeBuffer(_w[l], 0, segment.toJS);
      offset += count;
    }
  }

  @override
  Future<Float32List> getWeights() async {
    final staging = _weightsStaging!;
    final enc = device.createCommandEncoder();
    var offset = 0;
    for (var l = numLayers - 2; l >= 0; --l) {
      final bytes = _wCounts[l] * 4;
      enc.copyBufferToBuffer(_w[l], 0, staging, offset * 4, bytes);
      offset += _wCounts[l];
    }
    queue.submit(<GPUCommandBuffer>[enc.finish()].toJS);
    await staging.mapAsync(GPUMapMode.read).toDart;
    final view = staging.getMappedRange().toDart.asFloat32List();
    final result = Float32List.fromList(view);
    staging.unmap();
    return result;
  }

  @override
  Future<void> resetOptimizer() async {
    for (var l = 0; l < numLayers - 1; ++l) {
      final n = _wCounts[l];
      final delta0 = Float32List(n)..fillRange(0, n, _rDelta0);
      final zero = Float32List(n);
      queue.writeBuffer(_prevDelta[l], 0, delta0.toJS);
      queue.writeBuffer(_lastUpdate[l], 0, zero.toJS);
    }
    _globalLearnError = 1.0;
    _lastGlobalLearnError = 1.0;
    await queue.onSubmittedWorkDone().toDart;
  }

  @override
  Future<double> runEpoch(double target, double lr, double momentum) async {
    if (!_passesBuilt) return _globalLearnError;
    final errCount = _numSamples * _outputSize;

    // Pass 1: forward + backprop + gradient, then copy errBuf -> staging.
    final enc = device.createCommandEncoder();
    for (final p in _pass1) {
      final cp = enc.beginComputePass();
      cp.setPipeline(p.pipeline);
      cp.setBindGroup(0, p.bindGroup);
      cp.dispatchWorkgroups(p.dispatchX, p.dispatchY, 1);
      cp.end();
    }
    enc.copyBufferToBuffer(_errBuf!, 0, _errStaging!, 0, errCount * 4);
    queue.submit(<GPUCommandBuffer>[enc.finish()].toJS);

    await _errStaging!.mapAsync(GPUMapMode.read).toDart;
    final view = _errStaging!.getMappedRange().toDart.asFloat32List();
    var sumSq = 0.0;
    for (var i = 0; i < view.length; ++i) {
      sumSq += view[i];
    }
    _errStaging!.unmap();

    final globalError = sumSq / (_outputSize * _numSamples);
    _lastGlobalLearnError = _globalLearnError;
    _globalLearnError = globalError;

    if (globalError < target) {
      return globalError;
    }

    // Pass 2: weight update.
    final backtrack = _globalLearnError > _lastGlobalLearnError ? 1 : 0;
    for (var u = 0; u < _updatePasses.length; ++u) {
      final wCount = _updateWCounts[u];
      final params = _useRprop
          ? _params([backtrack, wCount], [_rEtaPlus, _rEtaMinus, _rMin, _rMax])
          : _params([wCount], [lr, momentum]);
      queue.writeBuffer(_updatePasses[u].uniform, 0, params);
    }

    final enc2 = device.createCommandEncoder();
    for (final p in _updatePasses) {
      final cp = enc2.beginComputePass();
      cp.setPipeline(p.pipeline);
      cp.setBindGroup(0, p.bindGroup);
      cp.dispatchWorkgroups(p.dispatchX, p.dispatchY, 1);
      cp.end();
    }
    queue.submit(<GPUCommandBuffer>[enc2.finish()].toJS);
    await queue.onSubmittedWorkDone().toDart;

    return globalError;
  }

  @override
  Future<Float32List> activate(Float32List input, int outputSize) async {
    final last = numLayers - 1;

    // Load input (+ bias) into outInf[0].
    final in0 = Float32List(sizes[0]);
    for (var i = 0; i < input.length && i < sizes[0]; ++i) {
      in0[i] = input[i];
    }
    if (spec.withBias[0]) in0[sizes[0] - 1] = 1.0;
    queue.writeBuffer(_outInf[0], 0, in0.toJS);

    final enc = device.createCommandEncoder();
    for (var l = 0; l < numLayers - 1; ++l) {
      final nextIsOutput = (l + 1) == last;
      final biasIndex = (!nextIsOutput && spec.withBias[l + 1])
          ? sizes[l + 1] - 1
          : -1;
      final pass = _makePass(
        _pForward,
        [_w[l], _outInf[l], _outInf[l + 1]],
        _params(
          [sizes[l], sizes[l + 1], spec.activationIds[l + 1], biasIndex, 1],
          [spec.activationScale[l + 1]],
        ),
        sizes[l + 1],
        1,
        8,
        8,
      );
      final cp = enc.beginComputePass();
      cp.setPipeline(pass.pipeline);
      cp.setBindGroup(0, pass.bindGroup);
      cp.dispatchWorkgroups(pass.dispatchX, pass.dispatchY, 1);
      cp.end();
    }
    enc.copyBufferToBuffer(_outInf[last], 0, _infStaging, 0, sizes[last] * 4);
    queue.submit(<GPUCommandBuffer>[enc.finish()].toJS);

    await _infStaging.mapAsync(GPUMapMode.read).toDart;
    final view = _infStaging.getMappedRange().toDart.asFloat32List();
    final result = Float32List(outputSize);
    for (var k = 0; k < outputSize && k < view.length; ++k) {
      result[k] = view[k];
    }
    _infStaging.unmap();
    return result;
  }

  @override
  void dispose() {
    void d(GPUBuffer b) => b.destroy();
    _w.forEach(d);
    _g.forEach(d);
    _gPrev.forEach(d);
    _prevDelta.forEach(d);
    _lastUpdate.forEach(d);
    _out.forEach(d);
    _delta.forEach(d);
    _outInf.forEach(d);
    _inputs?.destroy();
    _targets?.destroy();
    _errBuf?.destroy();
    _errStaging?.destroy();
    _weightsStaging?.destroy();
    _infStaging.destroy();
  }
}
