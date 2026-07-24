import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Second-order trainers (finite-difference based): Levenberg–Marquardt should
/// converge; Conjugate Gradient and L-BFGS should substantially reduce error.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(4, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  test('Levenberg–Marquardt converges on XOR', () {
    final t = LevenbergMarquardt(build(), SamplesSet(xor(), subject: 'xor'))
      ..logEnabled = false
      ..enableSelectInitialANN = false;
    final ok = t.trainUntilGlobalError(targetGlobalError: 1e-4, maxEpochs: 500);
    expect(ok, isTrue);
    for (final s in xor()) {
      t.ann.activate(s.input);
      expect(
        (t.ann.outputAsDouble.first - s.output.valuesAsDouble.first).abs(),
        lessThan(0.1),
      );
    }
  });

  test('Conjugate Gradient reduces error on XOR', () {
    final t = ConjugateGradient(build(), SamplesSet(xor(), subject: 'xor'))
      ..logEnabled = false;
    final before = t.ann.computeSamplesGlobalError(xor());
    t.train(200, 0.0);
    final after = t.ann.computeSamplesGlobalError(xor());
    expect(after, lessThan(before * 0.5));
  });

  test('L-BFGS reduces error on XOR', () {
    final t = LBFGS(build(), SamplesSet(xor(), subject: 'xor'))
      ..logEnabled = false;
    final before = t.ann.computeSamplesGlobalError(xor());
    t.train(200, 0.0);
    final after = t.ann.computeSamplesGlobalError(xor());
    expect(after, lessThan(before * 0.5));
  });
}
