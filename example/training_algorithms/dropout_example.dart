import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

/// Dropout: set a `dropoutRate` on a [HiddenLayerConfig]. Dropout is applied
/// (inverted) only while the trainer accumulates gradients; inference and error
/// evaluation run the full network.
///
/// Run: dart run example/training_algorithms/dropout_example.dart
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;
  final samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );
  final samplesSet = SamplesSet(samples, subject: 'xor');

  final ann = ANN(
    scale,
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    // 20% dropout on the hidden layer during training:
    [HiddenLayerConfig(6, true, ActivationFunctionSigmoid(), 0.2)],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
    random: Random(101),
  );

  final trainer = Adam(ann, samplesSet, learningRate: 0.05)..logEnabled = false;
  final before = ann.computeSamplesGlobalError(samples);
  trainer.train(3000, 0.0);
  final after = ann.computeSamplesGlobalError(samples);
  print(
    'Dropout(0.2) + Adam: '
    'error ${before.toStringAsExponential(3)} -> ${after.toStringAsExponential(3)}',
  );
  for (final s in samples) {
    ann.activate(s.input);
    print('  ${s.input.values} -> ${ann.outputAsDouble} (${s.output.values})');
  }
}
