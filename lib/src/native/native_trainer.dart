import 'dart:typed_data';

import '../eneural_net_activation_functions.dart';
import '../eneural_net_ann.dart';
import '../eneural_net_sample.dart';
import '../eneural_net_scale.dart';
import '../eneural_net_signal.dart';
import '../eneural_net_training_backpropagation.dart';
import '../eneural_net_training_propagation.dart';
import '../eneural_net_training_rprop.dart';
import 'native_accelerator.dart';
import 'native_backend.dart';

export 'native_backend.dart' show NativeBackend;

/// Signal type accepted by the native-accelerated trainers (Float32x4 only).
typedef NativeSample =
    Sample<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Maps an [ActivationFunction] to the native activation id, or `null` when the
/// function is not supported by the native backends.
int? _activationId(ActivationFunction af) {
  switch (af.name) {
    case 'Linear':
      return 0;
    case 'Sigmoid':
      return 1;
    case 'SigmoidFast':
      return 2;
    case 'SigmoidBoundedFast':
      return 3;
    default:
      return null;
  }
}

/// Builds a [NativeNetworkSpec] for [ann], or `null` if the network uses an
/// activation function or signal format not supported by the native backends.
NativeNetworkSpec? _buildSpec(
  ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
) {
  final format = ann.format;
  if (format != 'Float32x4' && format != 'Float32x4Mod4') return null;

  final layers = ann.allLayers;
  final neuronCounts = <int>[];
  final withBias = <bool>[];
  final activationIds = <int>[];
  final activationScale = <double>[];
  final flatSpots = <double>[];

  for (final layer in layers) {
    final af = layer.activationFunction;
    final id = _activationId(af);
    if (id == null) return null;

    neuronCounts.add(layer.length);
    withBias.add(layer.withBiasNeuron);
    activationIds.add(id);
    activationScale.add(
      af is ActivationFunctionSigmoidBoundedFast ? af.scale : 0.0,
    );
    flatSpots.add(af.flatSpot);
  }

  return NativeNetworkSpec(
    neuronCounts: neuronCounts,
    withBias: withBias,
    activationIds: activationIds,
    activationScale: activationScale,
    flatSpots: flatSpots,
  );
}

/// Shared native-acceleration logic for [NativeBackpropagation]/[NativeRProp].
///
/// When a native backend is resolved, [learn] runs a full epoch on device
/// (forward + backprop + weight update over the resident samples) and mirrors
/// the pure-Dart semantics: batch gradient accumulation, the iRProp+/backprop
/// update rules, and the early-exit-before-update behavior. If no backend is
/// available (web, unsupported activation/format, missing dylib) it transparently
/// falls back to the pure-Dart [Propagation.learn].
mixin _NativeTrainerMixin<P extends NativeSample>
    on Propagation<double, Float32x4, SignalFloat32x4, Scale<double>, P> {
  NativeAccelerator? _accelerator;
  bool _resolved = false;
  bool _samplesUploaded = false;

  Float32List? _weightsBuffer;

  /// Whether a native backend is active for this trainer.
  bool get isNativeAccelerated {
    _ensureResolved();
    return _accelerator != null;
  }

  /// The backend actually in use ([NativeBackend.none] when running pure-Dart).
  NativeBackend get activeBackend {
    _ensureResolved();
    return _accelerator?.backend ?? NativeBackend.none;
  }

  /// The requested backend (defaults to [NativeBackend.auto]).
  NativeBackend get requestedBackend => NativeBackend.auto;

  void _ensureResolved() {
    if (_resolved) return;
    _resolved = true;

    if (requestedBackend == NativeBackend.none) return;

    final spec = _buildSpec(ann);
    if (spec == null) return;

    final acc = resolveNativeAccelerator(
      requested: requestedBackend,
      spec: spec,
    );
    if (acc == null) return;

    // Configure the update rule.
    if (this is RProp) {
      acc.configRProp(0.10, RProp.weightMinStep, RProp.weightMaxStep, 1.2, 0.5);
    } else {
      acc.configBackprop();
    }

    _accelerator = acc;
    _weightsBuffer = Float32List(acc.weightsLength);
  }

  void _uploadSamples(List<P> samples, NativeAccelerator acc) {
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

    acc.setSamples(numSamples, inputSize, outputSize, inputs, targets);
    _samplesUploaded = true;
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    _ensureResolved();
    final acc = _accelerator;
    if (acc == null) {
      return super.learn(samples, targetGlobalError);
    }

    if (!_samplesUploaded) {
      _uploadSamples(samples, acc);
    }

    final weightsBuffer = _weightsBuffer!;

    // Upload current ANN weights (source of truth; may have been reset between
    // sessions). The optimizer state stays resident in native memory.
    final allWeights = ann.allWeights;
    for (var i = 0; i < weightsBuffer.length; ++i) {
      weightsBuffer[i] = allWeights[i];
    }
    acc.setWeights(weightsBuffer);

    final error = acc.runEpoch(targetGlobalError, learningRate, momentum);

    // Download the updated weights back into the ANN.
    acc.getWeights(weightsBuffer);
    ann.allWeights = weightsBuffer;

    // Publish the epoch error so BP learning-rate/momentum strategies adapt as
    // they would in the pure-Dart path (RProp uses static strategies).
    updateGlobalLearnError(error);

    return error < targetGlobalError;
  }

  /// Runs a native forward pass (inference) for [input] and returns the output
  /// layer values, or `null` when no native backend is active.
  ///
  /// The current ANN weights are uploaded first, so the result reflects the
  /// latest trained state.
  List<double>? activateNative(SignalFloat32x4 input) {
    _ensureResolved();
    final acc = _accelerator;
    if (acc == null) return null;

    final weightsBuffer = _weightsBuffer!;
    final allWeights = ann.allWeights;
    for (var i = 0; i < weightsBuffer.length; ++i) {
      weightsBuffer[i] = allWeights[i];
    }
    acc.setWeights(weightsBuffer);

    final inBuf = Float32List.fromList(input.valuesAsDouble);
    final outBuf = Float32List(ann.outputLayer.length);
    acc.activate(inBuf, outBuf);
    return outBuf.toList();
  }

  @override
  void reset() {
    super.reset();
    _accelerator?.resetOptimizer();
  }
}

/// Backpropagation trainer accelerated by a native CPU/Metal backend.
///
/// Drop-in replacement for [Backpropagation] on `Float32x4` networks. When no
/// native backend is available it behaves exactly like [Backpropagation].
class NativeBackpropagation<P extends NativeSample>
    extends
        Backpropagation<double, Float32x4, SignalFloat32x4, Scale<double>, P>
    with _NativeTrainerMixin<P> {
  final NativeBackend backend;

  NativeBackpropagation(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.backend = NativeBackend.auto,
    String? subject,
  }) : super(ann, samplesSet, subject: subject);

  @override
  NativeBackend get requestedBackend => backend;
}

/// Resilient Backpropagation (iRProp+) trainer accelerated by a native
/// CPU/Metal backend.
///
/// Drop-in replacement for [RProp] on `Float32x4` networks. When no native
/// backend is available it behaves exactly like [RProp].
class NativeRProp<P extends NativeSample>
    extends RProp<double, Float32x4, SignalFloat32x4, Scale<double>, P>
    with _NativeTrainerMixin<P> {
  final NativeBackend backend;

  NativeRProp(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.backend = NativeBackend.auto,
    String? subject,
    bool enableSelectInitialANN = false,
  }) : super(
         ann,
         samplesSet,
         subject: subject,
         enableSelectInitialANN: enableSelectInitialANN,
       );

  @override
  NativeBackend get requestedBackend => backend;
}
