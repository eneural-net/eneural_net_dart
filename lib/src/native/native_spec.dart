import 'dart:typed_data';

import '../eneural_net_activation_functions.dart';
import '../eneural_net_ann.dart';
import '../eneural_net_scale.dart';
import '../eneural_net_signal.dart';
import 'native_backend.dart';

/// Maps an [ActivationFunction] to the native/WebGPU activation id, or `null`
/// when the function is not supported by the accelerated backends.
///
/// 0=Linear, 1=Sigmoid, 2=SigmoidFast, 3=SigmoidBoundedFast.
int? activationIdOf(ActivationFunction af) {
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
/// activation function or signal format not supported by the accelerated
/// backends (CPU/Metal/WebGPU).
NativeNetworkSpec? buildNativeNetworkSpec(
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
    final id = activationIdOf(af);
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
