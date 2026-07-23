import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:path/path.dart' as p;

import 'native_backend.dart';

/// The Metal (GPU) backend has a fixed per-epoch overhead (~two command-buffer
/// syncs), so for small networks the CPU backend is faster. Empirically the
/// batched-GEMM Metal path overtakes CPU at roughly this many weights, so `auto`
/// switches to Metal at/above this threshold.
const int _metalAutoWeightsThreshold = 16384;

/// Resolves a [NativeAccelerator] for the requested backend, or `null` when no
/// native library is available for the current platform/architecture.
///
/// For `auto`, the CPU (Accelerate) backend is used for small networks and the
/// Metal (GPU) backend for large ones (see [_metalAutoWeightsThreshold]).
/// Request [NativeBackend.cpu] / [NativeBackend.metal] to force a backend.
NativeAccelerator? resolveNativeAccelerator({
  required NativeBackend requested,
  required NativeNetworkSpec spec,
}) {
  if (!Platform.isMacOS) return null;
  if (requested == NativeBackend.none) return null;

  final cpuAvailable = _CpuLibrary.instance.load();
  final metalAvailable = _MetalLibrary.instance.load();

  bool useMetal;
  switch (requested) {
    case NativeBackend.metal:
      useMetal = metalAvailable;
    case NativeBackend.cpu:
      useMetal = false;
    case NativeBackend.auto:
      useMetal =
          metalAvailable &&
          (!cpuAvailable || _weightsOf(spec) >= _metalAutoWeightsThreshold);
    case NativeBackend.none:
      return null;
  }

  if (useMetal && metalAvailable) {
    try {
      return _MetalAccelerator(spec);
    } catch (_) {
      // fall through to CPU
    }
  }

  if (cpuAvailable) {
    try {
      return _CpuAccelerator(spec);
    } catch (_) {
      return null;
    }
  }

  return null;
}

int _weightsOf(NativeNetworkSpec spec) {
  var total = 0;
  for (var l = 0; l < spec.numLayers - 1; ++l) {
    total += spec.neuronCounts[l] * spec.neuronCounts[l + 1];
  }
  return total;
}

// ============================================================
// Library loading
// ============================================================

String get _nativeArch =>
    Platform.version.contains('arm64') ? 'arm64' : 'x86_64';

/// Searches known locations for [fileNames] and returns the first existing one.
File? _findLibraryFile(List<String> fileNames) {
  final current = Directory.current.path;
  final candidates = <String>[
    current,
    p.join(current, 'native'),
    p.join(current, '..', 'eneural_net', 'native'),
    p.join(current, '..', 'eneural_net_dart', 'native'),
  ];

  for (final dir in candidates) {
    for (final name in fileNames) {
      final file = File(p.join(dir, name));
      if (file.existsSync()) return file.absolute;
    }
  }
  return null;
}

class _CpuLibrary {
  static final _CpuLibrary instance = _CpuLibrary._();
  _CpuLibrary._();

  bool? _loaded;

  bool load() {
    final cached = _loaded;
    if (cached != null) return cached;

    final arch = _nativeArch;
    final file = _findLibraryFile([
      'libeneural_cpu_$arch.dylib',
      p.join('macos', 'cpu', 'build', 'libeneural_cpu_$arch.dylib'),
    ]);

    if (file == null) return _loaded = false;

    try {
      DynamicLibrary.open(file.path);
      final available = _EnnCpu.available() == 1;
      return _loaded = available;
    } catch (_) {
      return _loaded = false;
    }
  }
}

class _MetalLibrary {
  static final _MetalLibrary instance = _MetalLibrary._();
  _MetalLibrary._();

  bool? _loaded;

  bool load() {
    final cached = _loaded;
    if (cached != null) return cached;

    final arch = _nativeArch;
    final file = _findLibraryFile([
      'libeneural_metal_$arch.dylib',
      p.join('macos', 'metal', 'build', 'libeneural_metal_$arch.dylib'),
    ]);

    if (file == null) return _loaded = false;

    try {
      DynamicLibrary.open(file.path);
      // Probe the device (fails gracefully under Rosetta / headless).
      final available = _EnnMetal.available() == 1;
      return _loaded = available;
    } catch (_) {
      return _loaded = false;
    }
  }
}

// ============================================================
// FFI bindings (symbols exported by libeneural_cpu_<arch>.dylib)
// ============================================================

class _EnnCpu {
  @Native<Int32 Function()>(symbol: 'enn_cpu_available', isLeaf: true)
  external static int available();

  @Native<
    Pointer<Void> Function(
      Int32,
      Pointer<Int32>,
      Pointer<Int32>,
      Pointer<Int32>,
      Pointer<Float>,
      Pointer<Float>,
    )
  >(symbol: 'enn_cpu_create_network', isLeaf: true)
  external static Pointer<Void> createNetwork(
    int numLayers,
    Pointer<Int32> neuronCounts,
    Pointer<Int32> withBias,
    Pointer<Int32> activationIds,
    Pointer<Float> activationScale,
    Pointer<Float> flatSpots,
  );

  @Native<Void Function(Pointer<Void>)>(symbol: 'enn_cpu_destroy', isLeaf: true)
  external static void destroy(Pointer<Void> net);

  @Native<Int32 Function(Pointer<Void>)>(
    symbol: 'enn_cpu_weights_length',
    isLeaf: true,
  )
  external static int weightsLength(Pointer<Void> net);

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32)>(
    symbol: 'enn_cpu_set_weights',
    isLeaf: true,
  )
  external static void setWeights(
    Pointer<Void> net,
    Pointer<Float> weights,
    int length,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32)>(
    symbol: 'enn_cpu_get_weights',
    isLeaf: true,
  )
  external static void getWeights(
    Pointer<Void> net,
    Pointer<Float> out,
    int length,
  );

  @Native<
    Void Function(
      Pointer<Void>,
      Int32,
      Int32,
      Int32,
      Pointer<Float>,
      Pointer<Float>,
    )
  >(symbol: 'enn_cpu_set_samples', isLeaf: true)
  external static void setSamples(
    Pointer<Void> net,
    int numSamples,
    int inputSize,
    int outputSize,
    Pointer<Float> inputs,
    Pointer<Float> targets,
  );

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'enn_cpu_config_backprop',
    isLeaf: true,
  )
  external static void configBackprop(Pointer<Void> net);

  @Native<Void Function(Pointer<Void>, Float, Float, Float, Float, Float)>(
    symbol: 'enn_cpu_config_rprop',
    isLeaf: true,
  )
  external static void configRProp(
    Pointer<Void> net,
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  );

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'enn_cpu_reset_optimizer',
    isLeaf: true,
  )
  external static void resetOptimizer(Pointer<Void> net);

  @Native<Float Function(Pointer<Void>, Float, Float, Float)>(
    symbol: 'enn_cpu_run_epoch',
    isLeaf: true,
  )
  external static double runEpoch(
    Pointer<Void> net,
    double target,
    double lr,
    double momentum,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Float>, Int32)
  >(symbol: 'enn_cpu_activate', isLeaf: true)
  external static void activate(
    Pointer<Void> net,
    Pointer<Float> input,
    int inputSize,
    Pointer<Float> output,
    int outputSize,
  );
}

// ============================================================
// CPU accelerator
// ============================================================

class _CpuAccelerator implements NativeAccelerator {
  Pointer<Void> _handle;
  bool _disposed = false;

  _CpuAccelerator._(this._handle);

  factory _CpuAccelerator(NativeNetworkSpec spec) {
    final counts = Int32List.fromList(spec.neuronCounts);
    final bias = Int32List.fromList(
      spec.withBias.map((e) => e ? 1 : 0).toList(),
    );
    final acts = Int32List.fromList(spec.activationIds);
    final scales = Float32List.fromList(spec.activationScale);
    final flats = Float32List.fromList(spec.flatSpots);

    final handle = _EnnCpu.createNetwork(
      spec.numLayers,
      counts.address,
      bias.address,
      acts.address,
      scales.address,
      flats.address,
    );

    if (handle == nullptr) {
      throw StateError('enn_cpu_create_network returned null');
    }
    return _CpuAccelerator._(handle);
  }

  @override
  NativeBackend get backend => NativeBackend.cpu;

  @override
  int get weightsLength => _EnnCpu.weightsLength(_handle);

  @override
  void setWeights(Float32List weights) =>
      _EnnCpu.setWeights(_handle, weights.address, weights.length);

  @override
  void getWeights(Float32List out) =>
      _EnnCpu.getWeights(_handle, out.address, out.length);

  @override
  void setSamples(
    int numSamples,
    int inputSize,
    int outputSize,
    Float32List inputs,
    Float32List targets,
  ) => _EnnCpu.setSamples(
    _handle,
    numSamples,
    inputSize,
    outputSize,
    inputs.address,
    targets.address,
  );

  @override
  void configBackprop() => _EnnCpu.configBackprop(_handle);

  @override
  void configRProp(
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  ) => _EnnCpu.configRProp(
    _handle,
    delta0,
    deltaMin,
    deltaMax,
    etaPlus,
    etaMinus,
  );

  @override
  void resetOptimizer() => _EnnCpu.resetOptimizer(_handle);

  @override
  double runEpoch(double target, double lr, double momentum) =>
      _EnnCpu.runEpoch(_handle, target, lr, momentum);

  @override
  void activate(Float32List input, Float32List output) => _EnnCpu.activate(
    _handle,
    input.address,
    input.length,
    output.address,
    output.length,
  );

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _EnnCpu.destroy(_handle);
    _handle = nullptr;
  }
}

// ============================================================
// FFI bindings (symbols exported by libeneural_metal_<arch>.dylib)
// ============================================================

class _EnnMetal {
  @Native<Int32 Function()>(symbol: 'enn_metal_available', isLeaf: true)
  external static int available();

  @Native<
    Pointer<Void> Function(
      Int32,
      Pointer<Int32>,
      Pointer<Int32>,
      Pointer<Int32>,
      Pointer<Float>,
      Pointer<Float>,
    )
  >(symbol: 'enn_metal_create_network', isLeaf: true)
  external static Pointer<Void> createNetwork(
    int numLayers,
    Pointer<Int32> neuronCounts,
    Pointer<Int32> withBias,
    Pointer<Int32> activationIds,
    Pointer<Float> activationScale,
    Pointer<Float> flatSpots,
  );

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'enn_metal_destroy',
    isLeaf: true,
  )
  external static void destroy(Pointer<Void> net);

  @Native<Int32 Function(Pointer<Void>)>(
    symbol: 'enn_metal_weights_length',
    isLeaf: true,
  )
  external static int weightsLength(Pointer<Void> net);

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32)>(
    symbol: 'enn_metal_set_weights',
    isLeaf: true,
  )
  external static void setWeights(
    Pointer<Void> net,
    Pointer<Float> weights,
    int length,
  );

  @Native<Void Function(Pointer<Void>, Pointer<Float>, Int32)>(
    symbol: 'enn_metal_get_weights',
    isLeaf: true,
  )
  external static void getWeights(
    Pointer<Void> net,
    Pointer<Float> out,
    int length,
  );

  @Native<
    Void Function(
      Pointer<Void>,
      Int32,
      Int32,
      Int32,
      Pointer<Float>,
      Pointer<Float>,
    )
  >(symbol: 'enn_metal_set_samples', isLeaf: true)
  external static void setSamples(
    Pointer<Void> net,
    int numSamples,
    int inputSize,
    int outputSize,
    Pointer<Float> inputs,
    Pointer<Float> targets,
  );

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'enn_metal_config_backprop',
    isLeaf: true,
  )
  external static void configBackprop(Pointer<Void> net);

  @Native<Void Function(Pointer<Void>, Float, Float, Float, Float, Float)>(
    symbol: 'enn_metal_config_rprop',
    isLeaf: true,
  )
  external static void configRProp(
    Pointer<Void> net,
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  );

  @Native<Void Function(Pointer<Void>)>(
    symbol: 'enn_metal_reset_optimizer',
    isLeaf: true,
  )
  external static void resetOptimizer(Pointer<Void> net);

  @Native<Float Function(Pointer<Void>, Float, Float, Float)>(
    symbol: 'enn_metal_run_epoch',
    isLeaf: true,
  )
  external static double runEpoch(
    Pointer<Void> net,
    double target,
    double lr,
    double momentum,
  );

  @Native<
    Void Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Float>, Int32)
  >(symbol: 'enn_metal_activate', isLeaf: true)
  external static void activate(
    Pointer<Void> net,
    Pointer<Float> input,
    int inputSize,
    Pointer<Float> output,
    int outputSize,
  );
}

// ============================================================
// Metal accelerator
// ============================================================

class _MetalAccelerator implements NativeAccelerator {
  Pointer<Void> _handle;
  bool _disposed = false;

  _MetalAccelerator._(this._handle);

  factory _MetalAccelerator(NativeNetworkSpec spec) {
    final counts = Int32List.fromList(spec.neuronCounts);
    final bias = Int32List.fromList(
      spec.withBias.map((e) => e ? 1 : 0).toList(),
    );
    final acts = Int32List.fromList(spec.activationIds);
    final scales = Float32List.fromList(spec.activationScale);
    final flats = Float32List.fromList(spec.flatSpots);

    final handle = _EnnMetal.createNetwork(
      spec.numLayers,
      counts.address,
      bias.address,
      acts.address,
      scales.address,
      flats.address,
    );

    if (handle == nullptr) {
      throw StateError('enn_metal_create_network returned null');
    }
    return _MetalAccelerator._(handle);
  }

  @override
  NativeBackend get backend => NativeBackend.metal;

  @override
  int get weightsLength => _EnnMetal.weightsLength(_handle);

  @override
  void setWeights(Float32List weights) =>
      _EnnMetal.setWeights(_handle, weights.address, weights.length);

  @override
  void getWeights(Float32List out) =>
      _EnnMetal.getWeights(_handle, out.address, out.length);

  @override
  void setSamples(
    int numSamples,
    int inputSize,
    int outputSize,
    Float32List inputs,
    Float32List targets,
  ) => _EnnMetal.setSamples(
    _handle,
    numSamples,
    inputSize,
    outputSize,
    inputs.address,
    targets.address,
  );

  @override
  void configBackprop() => _EnnMetal.configBackprop(_handle);

  @override
  void configRProp(
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  ) => _EnnMetal.configRProp(
    _handle,
    delta0,
    deltaMin,
    deltaMax,
    etaPlus,
    etaMinus,
  );

  @override
  void resetOptimizer() => _EnnMetal.resetOptimizer(_handle);

  @override
  double runEpoch(double target, double lr, double momentum) =>
      _EnnMetal.runEpoch(_handle, target, lr, momentum);

  @override
  void activate(Float32List input, Float32List output) => _EnnMetal.activate(
    _handle,
    input.address,
    input.length,
    output.address,
    output.length,
  );

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _EnnMetal.destroy(_handle);
    _handle = nullptr;
  }
}
