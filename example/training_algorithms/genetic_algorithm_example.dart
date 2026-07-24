import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

/// Genetic Algorithm — trains an XOR network.
///
/// Run: dart run example/training_algorithms/genetic_algorithm_example.dart
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
    [HiddenLayerConfig(6, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
    random: Random(101),
  );

  final trainer = GeneticAlgorithm(ann, samplesSet, random: Random(1))
    ..logEnabled = false;

  trainer.train(1000, 0.0);

  print(
    'Genetic Algorithm: '
    'error=${trainer.globalError.toStringAsExponential(3)} '
    'epochs=${trainer.totalTrainedEpochs}',
  );
  for (final s in samples) {
    ann.activate(s.input);
    print('  ${s.input.values} -> ${ann.outputAsDouble} (${s.output.values})');
  }
}
