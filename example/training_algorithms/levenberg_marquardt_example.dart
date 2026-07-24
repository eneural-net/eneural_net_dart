import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

/// Levenberg-Marquardt — trains an XOR network.
///
/// Run: dart run example/training_algorithms/levenberg_marquardt_example.dart
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

  final trainer = LevenbergMarquardt(ann, samplesSet)..logEnabled = false;

  // Damped Gauss-Newton converges in few iterations on small nets.
  trainer.train(100, 1e-5);

  print(
    'Levenberg-Marquardt: '
    'error=${trainer.globalError.toStringAsExponential(3)} '
    'epochs=${trainer.totalTrainedEpochs}',
  );
  for (final s in samples) {
    ann.activate(s.input);
    print('  ${s.input.values} -> ${ann.outputAsDouble} (${s.output.values})');
  }
}
