import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Mini-batch / online SGD tests. The batch loop shuffles samples and updates
/// once per mini-batch; a larger synthetic dataset is used so batching matters.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;
  final rnd = Random(7);

  // Synthetic linearly-separable-ish 4-input / 1-output regression set.
  List<SampleFloat32x4> dataset(int n) => List.generate(n, (_) {
    final input = List<double>.generate(4, (_) => rnd.nextDouble());
    final y = (input[0] + input[1] > input[2] + input[3]) ? 1.0 : 0.0;
    return SampleFloat32x4.fromNormalized(input, [y], scale);
  });

  ANNF build() => ANN(
    scale,
    LayerFloat32x4(4, true),
    [HiddenLayerConfig(8, true)],
    LayerFloat32x4(1, false),
    random: Random(11),
  );

  final samples = dataset(64);

  test('online SGD (batchSize 1) reduces error', () {
    final t = SGD(
      build(),
      SamplesSet(samples, subject: 'ds'),
      learningRate: 0.2,
      momentum: 0.9,
      batchSize: 1,
    )..logEnabled = false;

    final before = t.ann.computeSamplesGlobalError(samples);
    t.train(50, 0.0);
    final after = t.ann.computeSamplesGlobalError(samples);
    expect(after, lessThan(before));
  });

  test('mini-batch Adam (batchSize 16) reduces error', () {
    final t = Adam(
      build(),
      SamplesSet(samples, subject: 'ds'),
      learningRate: 0.02,
      batchSize: 16,
    )..logEnabled = false;

    final before = t.ann.computeSamplesGlobalError(samples);
    t.train(50, 0.0);
    final after = t.ann.computeSamplesGlobalError(samples);
    expect(after, lessThan(before));
  });
}
