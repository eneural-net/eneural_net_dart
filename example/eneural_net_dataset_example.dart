import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

/// Trains a neural network on a real, medium-size public dataset: the UCI
/// **Optical Recognition of Handwritten Digits** ("optdigits") set — 8x8
/// grayscale digits as 64 integer features (0..16), 10 classes, with an
/// official 3823-row train / 1797-row test split.
///
/// The two CSV files are downloaded once from the UCI repository and cached in
/// the system temp directory, then the network (64 -> 32 -> 10) is trained with
/// Adam and evaluated as test-set classification accuracy.
///
/// Run: dart run example/eneural_net_dataset_example.dart
Future<void> main() async {
  const base =
      'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/';
  final cacheDir = Directory(
    '${Directory.systemTemp.path}/eneural_net_optdigits',
  );

  print('Loading UCI optdigits dataset...');
  final String trainCsv;
  final String testCsv;
  try {
    trainCsv = await _fetch('${base}optdigits.tra', cacheDir, 'optdigits.tra');
    testCsv = await _fetch('${base}optdigits.tes', cacheDir, 'optdigits.tes');
  } catch (e) {
    stderr.writeln('\nCould not download the dataset: $e');
    stderr.writeln(
      'Download these two files manually into ${cacheDir.path}/ '
      'and re-run:\n  ${base}optdigits.tra\n  ${base}optdigits.tes',
    );
    exit(1);
  }

  final scale = ScaleDouble.ZERO_TO_ONE;
  final (trainSamples, trainLabels) = _parse(trainCsv, scale);
  final (testSamples, testLabels) = _parse(testCsv, scale);
  print(
    '  train: ${trainSamples.length} samples, '
    'test: ${testSamples.length} samples, '
    '${trainSamples.first.input.length} features, 10 classes\n',
  );

  // Network: 64 inputs -> 32 hidden (sigmoid) -> 10 outputs (sigmoid, one-hot).
  final ann = ANN(
    scale,
    LayerFloat32x4(64, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(32, true, ActivationFunctionSigmoid())],
    LayerFloat32x4(10, false, ActivationFunctionSigmoid()),
  );

  final trainer = Adam(
    ann,
    SamplesSet(trainSamples, subject: 'optdigits'),
    learningRate: 0.01,
    batchSize: 32,
  )..logEnabled = false;

  print(
    'Training ${trainer.algorithmName} (batchSize ${trainer.batchSize})...',
  );
  print('epoch |  train MSE | test acc | elapsed');
  print('------+------------+----------+--------');

  final sw = Stopwatch()..start();
  const blockEpochs = 2;
  const totalEpochs = 24;
  for (var e = blockEpochs; e <= totalEpochs; e += blockEpochs) {
    trainer.train(blockEpochs, 0.0);
    final mse = ann.computeSamplesGlobalError(trainSamples);
    final acc = _accuracy(ann, testSamples, testLabels);
    print(
      '${e.toString().padLeft(5)} | '
      '${mse.toStringAsExponential(3)} | '
      '${(acc * 100).toStringAsFixed(2).padLeft(6)}% | '
      '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)}s',
    );
  }
  sw.stop();

  final trainAcc = _accuracy(ann, trainSamples, trainLabels);
  final testAcc = _accuracy(ann, testSamples, testLabels);
  print(
    '\nFinal: train accuracy ${(trainAcc * 100).toStringAsFixed(2)}%, '
    'test accuracy ${(testAcc * 100).toStringAsFixed(2)}% '
    '(${trainer.totalTrainedEpochs} epochs in '
    '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)}s)',
  );
}

/// Downloads [url] into [dir]/[name], caching it; returns the file contents.
Future<String> _fetch(String url, Directory dir, String name) async {
  final file = File('${dir.path}/$name');
  if (file.existsSync() && file.lengthSync() > 0) {
    return file.readAsStringSync();
  }
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(Uri.parse(url))).close();
    if (response.statusCode != 200) {
      throw HttpException('HTTP ${response.statusCode} for $url');
    }
    final content = await response.transform(utf8.decoder).join();
    dir.createSync(recursive: true);
    file.writeAsStringSync(content);
    return content;
  } finally {
    client.close();
  }
}

/// Parses optdigits CSV rows: 64 features (0..16) + a class label (0..9).
/// Features are scaled to 0..1; labels are one-hot encoded into 10 outputs.
(List<SampleFloat32x4>, List<int>) _parse(String csv, Scale<double> scale) {
  final samples = <SampleFloat32x4>[];
  final labels = <int>[];
  for (final line in const LineSplitter().convert(csv)) {
    if (line.trim().isEmpty) continue;
    final parts = line.split(',');
    if (parts.length < 65) continue;
    final input = List<double>.generate(64, (i) => int.parse(parts[i]) / 16.0);
    final label = int.parse(parts[64]);
    final output = List<double>.filled(10, 0.0)..[label] = 1.0;
    samples.add(SampleFloat32x4.fromNormalized(input, output, scale));
    labels.add(label);
  }
  return (samples, labels);
}

/// Classification accuracy: argmax of the 10 outputs vs the true label.
double _accuracy(
  ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
  List<SampleFloat32x4> samples,
  List<int> labels,
) {
  var correct = 0;
  for (var i = 0; i < samples.length; ++i) {
    ann.activate(samples[i].input);
    final out = ann.outputAsDouble;
    var best = 0;
    for (var k = 1; k < out.length; ++k) {
      if (out[k] > out[best]) best = k;
    }
    if (best == labels[i]) correct++;
  }
  return correct / samples.length;
}
