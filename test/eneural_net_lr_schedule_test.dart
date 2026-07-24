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

  ANNF build() => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(4, true)],
    LayerFloat32x4(1, false),
    random: Random(101),
  );

  test('step decay lowers the learning rate over epochs', () {
    final t = SGD(
      build(),
      SamplesSet(xor()),
      learningRate: 0.5,
      lrSchedule: (p, base) =>
          StepDecayStrategy(p, base, stepSize: 20, gamma: 0.5),
    )..logEnabled = false;

    final lrStart = t.learningRate;
    t.train(60, 0.0);
    final lrLater = t.learningRate;

    expect(lrStart, closeTo(0.5, 1e-9));
    expect(lrLater, lessThan(lrStart));
  });

  test('cosine annealing + Adam converges on XOR', () {
    final t =
        Adam(
            build(),
            SamplesSet(xor()),
            learningRate: 0.05,
            lrSchedule: (p, base) => CosineAnnealingStrategy(
              p,
              base,
              maxEpochs: 1000,
              minValue: 0.001,
            ),
          )
          ..logEnabled = false
          ..enableSelectInitialANN = false;
    final ok = t.trainUntilGlobalError(
      targetGlobalError: 1e-3,
      maxEpochs: 5000,
    );
    expect(ok, isTrue);
  });
}
