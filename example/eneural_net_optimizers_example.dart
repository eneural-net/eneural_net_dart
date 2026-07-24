import 'dart:convert';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Demonstrates the training-algorithm library: a few optimizers, building a
/// trainer by name, and JSON checkpoint save/restore.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;
  final samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );
  final samplesSet = SamplesSet(samples, subject: 'xor');

  ANNF build() => ANN(
    scale,
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(4, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
  );

  // A handful of optimizers.
  for (final make in <TrainingD Function()>[
    () => Adam(build(), samplesSet, learningRate: 0.05),
    () => RMSProp(build(), samplesSet, learningRate: 0.02),
    () => Lion(build(), samplesSet, learningRate: 0.02),
    () => LevenbergMarquardt(build(), samplesSet),
    () => GeneticAlgorithm(build(), samplesSet),
  ]) {
    final t = make()
      ..logEnabled = false
      ..enableSelectInitialANN = false;
    t.trainUntilGlobalError(targetGlobalError: 1e-3, maxEpochs: 20000);
    print(
      '${t.algorithmName.padRight(20)} '
      'error=${t.globalError.toStringAsExponential(2)} '
      'epochs=${t.totalTrainedEpochs}',
    );
  }

  print('\nRegistered algorithms: ${registeredTrainings().join(', ')}\n');

  // Build by name + checkpoint round-trip.
  final trainer = trainingByName(
    'adamw',
    build(),
    samplesSet,
    params: {'learningRate': 0.05, 'weightDecay': 0.001},
  )..logEnabled = false;
  trainer.train(100, 0.0);

  final json = jsonEncode(saveTrainingCheckpoint(trainer));
  print(
    'Checkpoint (${json.length} bytes) after 100 epochs, '
    'error=${trainer.globalError.toStringAsExponential(2)}',
  );

  final resumed = trainingByName(
    'adamw',
    build(),
    samplesSet,
    params: {'learningRate': 0.05, 'weightDecay': 0.001},
  )..logEnabled = false;
  restoreTrainingCheckpoint(resumed, jsonDecode(json) as Map<String, dynamic>);
  resumed.train(100, 0.0);
  print(
    'Resumed +100 epochs -> error=${resumed.globalError.toStringAsExponential(2)}',
  );
}
