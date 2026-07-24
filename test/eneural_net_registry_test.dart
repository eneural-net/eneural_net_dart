import 'dart:convert';
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

  ANNF build({int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(4, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  test('trainingByName builds a working Adam trainer', () {
    final t =
        trainingByName(
            'adam',
            build(),
            SamplesSet(xor(), subject: 'xor'),
            params: {'learningRate': 0.05},
          )
          ..logEnabled = false
          ..enableSelectInitialANN = false;
    expect(t.algorithmName, equals('Adam'));
    final ok = t.trainUntilGlobalError(
      targetGlobalError: 1e-3,
      maxEpochs: 20000,
    );
    expect(ok, isTrue);
  });

  test('unknown algorithm throws', () {
    expect(
      () => trainingByName('nope', build(), SamplesSet(xor())),
      throwsStateError,
    );
  });

  test('registry lists many algorithms', () {
    expect(registeredTrainings().length, greaterThan(15));
    expect(registeredTrainings(), contains('adam'));
    expect(registeredTrainings(), contains('levenbergmarquardt'));
  });

  test('checkpoint resume matches uninterrupted training (Adam)', () {
    // Uninterrupted: 200 epochs.
    final ref = Adam(build(), SamplesSet(xor()), learningRate: 0.03)
      ..logEnabled = false;
    ref.train(200, 0.0);
    final refWeights = ref.ann.allWeights;

    // Interrupted: 100 epochs -> checkpoint (round-tripped through JSON) -> new
    // trainer restores it -> 100 more epochs.
    final first = Adam(build(), SamplesSet(xor()), learningRate: 0.03)
      ..logEnabled = false;
    first.train(100, 0.0);
    final checkpoint =
        jsonDecode(jsonEncode(saveTrainingCheckpoint(first)))
            as Map<String, dynamic>;

    final resumed = Adam(
      build(seed: 999),
      SamplesSet(xor()),
      learningRate: 0.03,
    )..logEnabled = false;
    restoreTrainingCheckpoint(resumed, checkpoint);
    resumed.train(100, 0.0);
    final resumedWeights = resumed.ann.allWeights;

    var maxDiff = 0.0;
    for (var i = 0; i < refWeights.length; ++i) {
      final d = (refWeights[i] - resumedWeights[i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    expect(maxDiff, lessThan(1e-5), reason: 'resumed run should match');
  });
}
