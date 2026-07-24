import 'dart:convert';

import 'package:eneural_net/eneural_net.dart';

import 'common.dart';

/// Multi-class classification on the UCI **Optical Recognition of Handwritten
/// Digits** ("optdigits") set: 8x8 grayscale digits as 64 features (0..16),
/// 10 classes, official 3823-row train / 1797-row test split.
///
/// Network 64 -> 32 -> 10, trained with iRProp+. The whole-epoch training can be
/// accelerated by a native backend (see [parseBackend]):
///
///   dart run example/datasets/optdigits_example.dart [none|auto|cpu|metal]
Future<void> main(List<String> args) async {
  final backend = parseBackend(args);
  const base =
      'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/';

  print('Loading UCI optdigits...');
  final String trainCsv, testCsv;
  try {
    trainCsv = await fetchDataset('${base}optdigits.tra', 'optdigits.tra');
    testCsv = await fetchDataset('${base}optdigits.tes', 'optdigits.tes');
  } catch (e) {
    print('Could not download the dataset: $e');
    return;
  }

  final scale = ScaleDouble.ZERO_TO_ONE;
  final (trainSamples, trainLabels) = _parse(trainCsv, scale);
  final (testSamples, testLabels) = _parse(testCsv, scale);
  print(
    '  train ${trainSamples.length} / test ${testSamples.length}, '
    '64 features, 10 classes\n',
  );

  final ann = ANN(
    scale,
    LayerFloat32x4(64, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(32, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(10, false, ActivationFunctionSigmoid()),
  );

  final trainer = NativeRProp(
    ann,
    SamplesSet(trainSamples, subject: 'optdigits'),
    backend: backend,
  )..logEnabled = false;

  print(
    'Training ${trainer.algorithmName} '
    '(backend: requested ${args.isEmpty ? "none" : args.first}, '
    'active ${trainer.activeBackend.name})',
  );
  print('epoch |  train MSE | test acc | elapsed');
  print('------+------------+----------+--------');

  final sw = Stopwatch()..start();
  for (var e = 10; e <= 100; e += 10) {
    trainer.train(10, 0.0);
    final mse = ann.computeSamplesGlobalError(trainSamples);
    final acc = classificationAccuracy(ann, testSamples, testLabels);
    print(
      '${e.toString().padLeft(5)} | ${mse.toStringAsExponential(3)} | '
      '${(acc * 100).toStringAsFixed(2).padLeft(6)}% | '
      '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)}s',
    );
  }
  sw.stop();

  final trainAcc = classificationAccuracy(ann, trainSamples, trainLabels);
  final testAcc = classificationAccuracy(ann, testSamples, testLabels);
  print(
    '\nFinal: train ${(trainAcc * 100).toStringAsFixed(2)}%, '
    'test ${(testAcc * 100).toStringAsFixed(2)}%',
  );
}

(List<SampleFloat32x4>, List<int>) _parse(String csv, Scale<double> scale) {
  final samples = <SampleFloat32x4>[];
  final labels = <int>[];
  for (final line in const LineSplitter().convert(csv)) {
    if (line.trim().isEmpty) continue;
    final parts = line.split(',');
    if (parts.length < 65) continue;
    final input = List<double>.generate(64, (i) => int.parse(parts[i]) / 16.0);
    final label = int.parse(parts[64]);
    samples.add(
      SampleFloat32x4.fromNormalized(
        input,
        List<double>.filled(10, 0.0)..[label] = 1.0,
        scale,
      ),
    );
    labels.add(label);
  }
  return (samples, labels);
}
