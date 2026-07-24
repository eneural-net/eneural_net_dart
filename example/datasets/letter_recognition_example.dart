import 'dart:convert';
import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

import 'common.dart';

/// Larger multi-class classification on the UCI **Letter Recognition** dataset:
/// 20,000 samples, 16 integer features derived from images of the 26 capital
/// letters (A–Z), split 80/20 into train/test.
///
/// Features are min-max normalized per column; the letter is one-hot encoded
/// into 26 outputs. Network 16 -> 40 -> 26, trained with iRProp+.
///
///   dart run example/datasets/letter_recognition_example.dart [none|auto|cpu|metal]
Future<void> main(List<String> args) async {
  final backend = parseBackend(args);
  const url =
      'https://archive.ics.uci.edu/ml/machine-learning-databases/'
      'letter-recognition/letter-recognition.data';

  print('Loading UCI Letter Recognition...');
  final rows = <List<double>>[]; // 16 features
  final labels = <int>[]; // 0..25
  try {
    _parseInto(
      await fetchDataset(url, 'letter-recognition.data'),
      rows,
      labels,
    );
  } catch (e) {
    print('Could not download the dataset: $e');
    return;
  }

  normalizeColumns(rows);

  final order = List<int>.generate(rows.length, (i) => i)..shuffle(Random(42));
  final split = (rows.length * 0.8).floor();
  final scale = ScaleDouble.ZERO_TO_ONE;

  SampleFloat32x4 sampleAt(int i) => SampleFloat32x4.fromNormalized(
    rows[i],
    List<double>.filled(26, 0.0)..[labels[i]] = 1.0,
    scale,
  );
  final trainSamples = [for (var k = 0; k < split; ++k) sampleAt(order[k])];
  final trainLabels = [for (var k = 0; k < split; ++k) labels[order[k]]];
  final testSamples = [
    for (var k = split; k < order.length; ++k) sampleAt(order[k]),
  ];
  final testLabels = [
    for (var k = split; k < order.length; ++k) labels[order[k]],
  ];
  print(
    '  train ${trainSamples.length} / test ${testSamples.length}, '
    '16 features, 26 classes\n',
  );

  final ann = ANN(
    scale,
    LayerFloat32x4(16, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(40, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(26, false, ActivationFunctionSigmoid()),
  );

  final trainer = NativeRProp(
    ann,
    SamplesSet(trainSamples, subject: 'letters'),
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
  for (var e = 20; e <= 200; e += 20) {
    trainer.train(20, 0.0);
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

void _parseInto(String csv, List<List<double>> rows, List<int> labels) {
  for (final line in const LineSplitter().convert(csv)) {
    final l = line.trim();
    if (l.isEmpty) continue;
    final parts = l.split(',');
    if (parts.length < 17) continue;
    labels.add(parts[0].codeUnitAt(0) - 'A'.codeUnitAt(0)); // 'A'..'Z' -> 0..25
    rows.add([for (var j = 1; j <= 16; ++j) double.parse(parts[j])]);
  }
}
