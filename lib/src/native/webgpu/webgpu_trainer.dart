import 'dart:math';
import 'dart:typed_data';

import '../../eneural_net_ann.dart';
import '../../eneural_net_sample.dart';
import '../../eneural_net_scale.dart';
import '../../eneural_net_signal.dart';
import '../../eneural_net_training_backpropagation.dart';
import '../../eneural_net_training_propagation.dart';
import '../../eneural_net_training_rprop.dart';
import '../native_spec.dart';
import 'webgpu_accelerator.dart';
import 'webgpu_resolver.dart';

/// Signal type accepted by the WebGPU trainers (Float32x4 only).
typedef WebGpuSample =
    Sample<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Shared WebGPU async-training logic for the [WebGpuRProp]/[WebGpuBackpropagation]
/// trainers.
///
/// WebGPU is asynchronous, so training is driven by `Future`-returning methods
/// ([trainAsync], [trainUntilGlobalErrorAsync]). The whole epoch (forward +
/// backprop + weight update over every resident sample) runs on the GPU,
/// mirroring the pure-Dart iRProp+/Backpropagation numerics.
///
/// When WebGPU is unavailable (not a browser, no WebGPU support, unsupported
/// network) the async methods transparently run the synchronous pure-Dart
/// trainer instead.
mixin WebGpuTrainerMixin<P extends WebGpuSample>
    on Propagation<double, Float32x4, SignalFloat32x4, Scale<double>, P> {
  WebGpuAccelerator? _acc;
  bool _resolved = false;
  bool _samplesUploaded = false;
  Float32List? _weightsBuffer;

  Future<void> _ensureResolved() async {
    if (_resolved) return;
    _resolved = true;

    final spec = buildNativeNetworkSpec(ann);
    if (spec == null) return;

    final acc = await resolveWebGpuAccelerator(
      spec: spec,
      rprop: this is RProp,
    );
    if (acc == null) return;

    if (this is RProp) {
      acc.configRProp(0.10, RProp.weightMinStep, RProp.weightMaxStep, 1.2, 0.5);
    } else {
      acc.configBackprop();
    }

    _acc = acc;
    _weightsBuffer = Float32List(acc.weightsLength);
  }

  /// Whether a WebGPU backend is active for this trainer (resolves it lazily).
  Future<bool> get isWebGpuAccelerated async {
    await _ensureResolved();
    return _acc != null;
  }

  Future<void> _uploadSamples(WebGpuAccelerator acc) async {
    final samples = this.samples;
    final numSamples = samples.length;
    final inputSize = samples.first.input.length;
    final outputSize = samples.first.output.length;

    final inputs = Float32List(numSamples * inputSize);
    final targets = Float32List(numSamples * outputSize);

    for (var s = 0; s < numSamples; ++s) {
      final inValues = samples[s].input.valuesAsDouble;
      final outValues = samples[s].output.valuesAsDouble;
      for (var i = 0; i < inputSize; ++i) {
        inputs[s * inputSize + i] = inValues[i];
      }
      for (var k = 0; k < outputSize; ++k) {
        targets[s * outputSize + k] = outValues[k];
      }
    }

    await acc.setSamples(numSamples, inputSize, outputSize, inputs, targets);
    _samplesUploaded = true;
  }

  void _writeWeightsBufferFromAnn() {
    final wb = _weightsBuffer!;
    final all = ann.allWeights;
    for (var i = 0; i < wb.length; ++i) {
      wb[i] = all[i];
    }
  }

  /// Runs a block of [epochs] on the GPU: uploads the current ANN weights, runs
  /// the epochs, then downloads the weights back into the ANN.
  ///
  /// The GPU never early-stops (a `0.0` target); convergence is decided by the
  /// callers via [Training.computeGlobalError] (the same metric as the sync
  /// path), avoiding an early-stop metric mismatch.
  Future<void> _runEpochBlock(WebGpuAccelerator acc, int epochs) async {
    if (!_samplesUploaded) await _uploadSamples(acc);

    _writeWeightsBufferFromAnn();
    await acc.setWeights(_weightsBuffer!);

    for (var e = 0; e < epochs; ++e) {
      final error = await acc.runEpoch(0.0, learningRate, momentum);
      updateGlobalLearnError(error);
      updateParameters();
    }

    final out = await acc.getWeights();
    ann.allWeights = out;
  }

  /// Trains [epochs] on the GPU (or the pure-Dart fallback) and returns the
  /// resulting global error. Weights are written back into the ANN.
  Future<double> trainAsync(int epochs, double targetGlobalError) async {
    initializeTraining();
    await _ensureResolved();
    final acc = _acc;
    if (acc == null) {
      return train(epochs, targetGlobalError);
    }

    await _runEpochBlock(acc, epochs);

    final globalError = computeGlobalError(samples);
    recordTrainingBlock(globalError, epochs);
    return globalError;
  }

  /// GPU-accelerated equivalent of [Training.trainUntilGlobalError].
  ///
  /// Falls back to the synchronous pure-Dart [Training.trainUntilGlobalError]
  /// when WebGPU is unavailable.
  Future<bool> trainUntilGlobalErrorAsync({
    double? targetGlobalError,
    int epochsBlock = 50,
    int maxEpochs = 1000000,
    int maxRetries = 3,
    Random? random,
  }) async {
    initializeTraining();
    targetGlobalError ??= samplesSet.targetGlobalError;

    await _ensureResolved();
    final acc = _acc;
    if (acc == null) {
      return trainUntilGlobalError(
        targetGlobalError: targetGlobalError,
        epochsBlock: epochsBlock,
        maxEpochs: maxEpochs,
        maxRetries: maxRetries,
        random: random,
      );
    }

    if (epochsBlock < 1) epochsBlock = 1;
    if (maxEpochs < 1) maxEpochs = 1;
    if (maxRetries < 0) maxRetries = 0;

    resetBestTraining();

    for (var retry = 0; retry <= maxRetries; ++retry) {
      await acc.resetOptimizer();
      updateGlobalLearnError(1.0);

      var trained = 0;

      while (trained < maxEpochs) {
        final block = min(epochsBlock, maxEpochs - trained);
        await _runEpochBlock(acc, block);
        trained += block;

        final globalError = computeGlobalError(samples);
        recordTrainingBlock(globalError, block);

        if (globalError <= targetGlobalError) return true;
      }

      // Retry with fresh random weights.
      if (retry < maxRetries) ann.resetWeights(random);
    }

    return false;
  }

  /// Runs a forward pass (inference) on the GPU for [input], or `null` when no
  /// WebGPU backend is active.
  Future<List<double>?> activateWebGpu(SignalFloat32x4 input) async {
    await _ensureResolved();
    final acc = _acc;
    if (acc == null) return null;

    _writeWeightsBufferFromAnn();
    await acc.setWeights(_weightsBuffer!);

    final result = await acc.activate(
      Float32List.fromList(input.valuesAsDouble),
      ann.outputLayer.length,
    );
    return result.toList();
  }
}

/// Backpropagation trainer accelerated by WebGPU (browser GPU).
///
/// Drop-in extension of [Backpropagation] on `Float32x4` networks adding the
/// async [trainAsync]/[trainUntilGlobalErrorAsync] methods. When WebGPU is
/// unavailable these fall back to the synchronous pure-Dart trainer.
class WebGpuBackpropagation<P extends WebGpuSample>
    extends
        Backpropagation<double, Float32x4, SignalFloat32x4, Scale<double>, P>
    with WebGpuTrainerMixin<P> {
  WebGpuBackpropagation(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    String? subject,
  }) : super(ann, samplesSet, subject: subject);
}

/// Resilient Backpropagation (iRProp+) trainer accelerated by WebGPU.
///
/// Drop-in extension of [RProp] on `Float32x4` networks adding the async
/// [trainAsync]/[trainUntilGlobalErrorAsync] methods. When WebGPU is unavailable
/// these fall back to the synchronous pure-Dart trainer.
class WebGpuRProp<P extends WebGpuSample>
    extends RProp<double, Float32x4, SignalFloat32x4, Scale<double>, P>
    with WebGpuTrainerMixin<P> {
  WebGpuRProp(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    String? subject,
    bool enableSelectInitialANN = false,
  }) : super(
         ann,
         samplesSet,
         subject: subject,
         enableSelectInitialANN: enableSelectInitialANN,
       );
}
