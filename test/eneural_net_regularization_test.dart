import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({double dropout = 0.0, int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(6, true, null, dropout)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  double weightNorm(ANNF ann) {
    var s = 0.0;
    for (final w in ann.allWeights) {
      s += w * w;
    }
    return sqrt(s);
  }

  test('L2 weight decay shrinks the weight norm', () {
    final plain = Adam(build(), SamplesSet(xor()), learningRate: 0.05)
      ..logEnabled = false;
    final decayed = Adam(
      build(),
      SamplesSet(xor()),
      learningRate: 0.05,
      weightDecay: 0.05,
    )..logEnabled = false;

    plain.train(400, 0.0);
    decayed.train(400, 0.0);

    expect(weightNorm(decayed.ann), lessThan(weightNorm(plain.ann)));
  });

  test('gradient clipping does not break convergence', () {
    final t =
        Adam(build(), SamplesSet(xor()), learningRate: 0.05, gradientClip: 0.5)
          ..logEnabled = false
          ..enableSelectInitialANN = false;
    final ok = t.trainUntilGlobalError(
      targetGlobalError: 1e-3,
      maxEpochs: 20000,
    );
    expect(ok, isTrue);
  });

  group('Dropout', () {
    test('network with dropout still learns XOR', () {
      final t = Adam(build(dropout: 0.2), SamplesSet(xor()), learningRate: 0.05)
        ..logEnabled = false;
      final before = t.ann.computeSamplesGlobalError(xor());
      t.train(3000, 0.0);
      final after = t.ann.computeSamplesGlobalError(xor());
      expect(after, lessThan(before * 0.5));
    });

    test('inference is deterministic (dropout off outside training)', () {
      final ann = build(dropout: 0.5);
      final input = xor().first.input;
      ann.activate(input);
      final out1 = List<double>.of(ann.outputAsDouble);
      ann.activate(input);
      final out2 = List<double>.of(ann.outputAsDouble);
      expect(out1, equals(out2));
      expect(ann.trainingMode, isFalse);
    });
  });
}
