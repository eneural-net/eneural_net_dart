import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

/// Learning-rate schedules: pass an `lrSchedule` builder to any
/// [GradientOptimizer]. Available: [StepDecayStrategy],
/// [ExponentialDecayStrategy], [CosineAnnealingStrategy], [WarmupStrategy].
///
/// Run: dart run example/training_algorithms/lr_schedule_example.dart
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
    [HiddenLayerConfig(3, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
    random: Random(101),
  );

  final trainer =
      Adam(
          ann,
          samplesSet,
          learningRate: 0.05,
          lrSchedule: (p, base) => CosineAnnealingStrategy(
            p,
            base,
            maxEpochs: 1000,
            minValue: 0.001,
          ),
        )
        ..logEnabled = false
        ..enableSelectInitialANN = false;

  print('lr @ start = ${trainer.learningRate.toStringAsFixed(4)}');
  trainer.trainUntilGlobalError(targetGlobalError: 1e-4, maxEpochs: 5000);
  print(
    'Cosine-annealed Adam: '
    'error=${trainer.globalError.toStringAsExponential(3)} '
    'epochs=${trainer.totalTrainedEpochs} '
    'lr@end=${trainer.learningRate.toStringAsExponential(2)}',
  );
}
