import 'dart:convert';
import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

import 'common.dart';

/// **Regression** on the UCI **Wine Quality** dataset (red + white combined):
/// 11 physicochemical features predict a wine-quality score (0..10). ~6,497
/// rows, split 80/20 into train/test.
///
/// Features are min-max normalized per column; the quality target is scaled to
/// 0..1 (÷10) and predicted by a single sigmoid output. Network 11 -> 16 -> 1,
/// trained with iRProp+; reported as RMSE and ±0.5 accuracy on the 0..10 scale.
///
///   dart run example/datasets/wine_quality_example.dart [none|auto|cpu|metal]
Future<void> main(List<String> args) async {
  final backend = parseBackend(args);
  const base =
      'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/';

  print('Loading UCI Wine Quality (red + white)...');
  final rows = <List<double>>[]; // 11 features
  final quality = <double>[]; // 0..10
  try {
    for (final name in ['winequality-red.csv', 'winequality-white.csv']) {
      _parseInto(await fetchDataset('$base$name', name), rows, quality);
    }
  } catch (e) {
    print('Could not download the dataset: $e');
    return;
  }

  normalizeColumns(rows);

  // Deterministic 80/20 shuffle-split.
  final order = List<int>.generate(rows.length, (i) => i)..shuffle(Random(42));
  final split = (rows.length * 0.8).floor();
  final scale = ScaleDouble.ZERO_TO_ONE;

  SampleFloat32x4 sampleAt(int i) =>
      SampleFloat32x4.fromNormalized(rows[i], [quality[i] / 10.0], scale);
  final train = [for (var k = 0; k < split; ++k) sampleAt(order[k])];
  final test = [for (var k = split; k < order.length; ++k) sampleAt(order[k])];
  final testQuality = [
    for (var k = split; k < order.length; ++k) quality[order[k]],
  ];
  print('  train ${train.length} / test ${test.length}, 11 features\n');

  final ann = ANN(
    scale,
    LayerFloat32x4(11, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(16, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
  );

  final trainer = NativeRProp(
    ann,
    SamplesSet(train, subject: 'wine'),
    backend: backend,
  )..logEnabled = false;

  print(
    'Training ${trainer.algorithmName} '
    '(backend: requested ${args.isEmpty ? "none" : args.first}, '
    'active ${trainer.activeBackend.name})',
  );
  print('epoch |  train MSE | test RMSE | ±0.5 acc | elapsed');
  print('------+------------+-----------+----------+--------');

  final sw = Stopwatch()..start();
  for (var e = 20; e <= 200; e += 20) {
    trainer.train(20, 0.0);
    final mse = ann.computeSamplesGlobalError(train);
    final (rmse, acc) = _evaluate(ann, test, testQuality);
    print(
      '${e.toString().padLeft(5)} | ${mse.toStringAsExponential(3)} | '
      '${rmse.toStringAsFixed(4).padLeft(9)} | '
      '${(acc * 100).toStringAsFixed(2).padLeft(6)}% | '
      '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)}s',
    );
  }
  sw.stop();

  final (rmse, acc) = _evaluate(ann, test, testQuality);
  print(
    '\nFinal: test RMSE ${rmse.toStringAsFixed(4)} (quality 0..10), '
    '±0.5 accuracy ${(acc * 100).toStringAsFixed(2)}%',
  );
}

void _parseInto(String csv, List<List<double>> rows, List<double> quality) {
  final lines = const LineSplitter().convert(csv);
  for (var i = 1; i < lines.length; ++i) {
    // skip header
    final line = lines[i].trim();
    if (line.isEmpty) continue;
    final parts = line.split(';');
    if (parts.length < 12) continue;
    rows.add([for (var j = 0; j < 11; ++j) double.parse(parts[j])]);
    quality.add(double.parse(parts[11]));
  }
}

/// Returns `(RMSE, ±0.5 accuracy)` on the 0..10 quality scale.
(double, double) _evaluate(
  ANNF ann,
  List<SampleFloat32x4> test,
  List<double> q,
) {
  var se = 0.0;
  var within = 0;
  for (var i = 0; i < test.length; ++i) {
    ann.activate(test[i].input);
    final pred = ann.outputAsDouble.first * 10.0;
    final err = pred - q[i];
    se += err * err;
    if (err.abs() <= 0.5) within++;
  }
  return (sqrt(se / test.length), within / test.length);
}
