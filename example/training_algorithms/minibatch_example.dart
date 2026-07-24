import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

/// Mini-batch / online training: any [GradientOptimizer] accepts a `batchSize`
/// (0 = full-batch, 1 = online SGD). Samples are shuffled each epoch.
///
/// Run: dart run example/training_algorithms/minibatch_example.dart
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;
  final rnd = Random(7);
  final samples = List.generate(64, (_) {
    final input = List<double>.generate(4, (_) => rnd.nextDouble());
    final y = (input[0] + input[1] > input[2] + input[3]) ? 1.0 : 0.0;
    return SampleFloat32x4.fromNormalized(input, [y], scale);
  });
  final samplesSet = SamplesSet(samples, subject: 'ds');

  final ann = ANN(
    scale,
    LayerFloat32x4(4, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(8, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
    random: Random(11),
  );

  final trainer = Adam(ann, samplesSet, learningRate: 0.02, batchSize: 16)
    ..logEnabled = false;

  final before = ann.computeSamplesGlobalError(samples);
  trainer.train(100, 0.0);
  final after = ann.computeSamplesGlobalError(samples);
  print(
    'Mini-batch Adam (batchSize 16): '
    'error ${before.toStringAsExponential(3)} -> ${after.toStringAsExponential(3)}',
  );
}
