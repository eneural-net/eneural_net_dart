import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;
typedef TrainerF =
    Training<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

/// Fills test/integration-coverage gaps across the training algorithms:
///  - every registered algorithm can be built and stepped (registry builders);
///  - JSON checkpoint round-trip resume for several stateful optimizers;
///  - the remaining LR schedules;
///  - a training-lifecycle / persistence "integration" pass over representative
///    new algorithms (bookkeeping, JSON round-trip + keep-training, NaN-free).
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

  group('Registry: every registered algorithm builds and steps', () {
    for (final name in registeredTrainings()) {
      test(name, () {
        final t = trainingByName(name, build(), SamplesSet(xor(), subject: 'x'))
          ..logEnabled = false
          ..enableSelectInitialANN = false;
        expect(t.algorithmName, isNotEmpty);
        t.train(10, 0.0);
        expect(t.globalError.isFinite, isTrue, reason: '$name error finite');
        expect(t.globalError.isNaN, isFalse);
      });
    }
  });

  group('Checkpoint resume matches uninterrupted training', () {
    // Every gradient optimizer must resume exactly — including Quickprop and
    // iRProp+, which read the previous gradient and (iRProp+) the epoch error
    // tracking, both now captured by the checkpoint.
    final makers = <String, TrainerF Function(ANNF)>{
      'Adam': (a) => Adam(a, SamplesSet(xor()), learningRate: 0.03),
      'RMSProp': (a) => RMSProp(a, SamplesSet(xor()), learningRate: 0.02),
      'AdaGrad': (a) => AdaGrad(a, SamplesSet(xor()), learningRate: 0.1),
      'AdaDelta': (a) => AdaDelta(a, SamplesSet(xor())),
      'Lion': (a) => Lion(a, SamplesSet(xor()), learningRate: 0.02),
      'SGD+M': (a) =>
          SGD(a, SamplesSet(xor()), learningRate: 0.3, momentum: 0.9),
      'Quickprop': (a) => Quickprop(a, SamplesSet(xor()), learningRate: 0.2),
      'iRProp+': (a) => ResilientPropagation(a, SamplesSet(xor())),
    };

    makers.forEach((name, make) {
      test(name, () {
        final ref = make(build())..logEnabled = false;
        ref.train(200, 0.0);
        final refWeights = ref.ann.allWeights;

        final first = make(build())..logEnabled = false;
        first.train(100, 0.0);
        final checkpoint =
            jsonDecode(jsonEncode(saveTrainingCheckpoint(first)))
                as Map<String, dynamic>;

        final resumed = make(build(seed: 999))..logEnabled = false;
        restoreTrainingCheckpoint(resumed, checkpoint);
        resumed.train(100, 0.0);

        var maxDiff = 0.0;
        final rw = resumed.ann.allWeights;
        for (var i = 0; i < refWeights.length; ++i) {
          final d = (refWeights[i] - rw[i]).abs();
          if (d > maxDiff) maxDiff = d;
        }
        expect(maxDiff, lessThan(1e-4), reason: '$name resume mismatch');
      });
    });
  });

  group('Learning-rate schedules', () {
    test('exponential decay lowers the value over epochs', () {
      final t = SGD(
        build(),
        SamplesSet(xor()),
        learningRate: 0.5,
        lrSchedule: (p, base) => ExponentialDecayStrategy(p, base, gamma: 0.9),
      )..logEnabled = false;
      final lrStart = t.learningRate;
      t.train(30, 0.0);
      expect(lrStart, closeTo(0.5, 1e-9));
      expect(t.learningRate, lessThan(lrStart));
    });

    test('warmup raises the value to base over warmupEpochs', () {
      final t = SGD(
        build(),
        SamplesSet(xor()),
        learningRate: 0.5,
        lrSchedule: (p, base) => WarmupStrategy(p, base, warmupEpochs: 20),
      )..logEnabled = false;
      final lrStart = t.learningRate; // epoch 0 -> base/20
      t.train(30, 0.0);
      expect(lrStart, lessThan(0.5));
      expect(t.learningRate, closeTo(0.5, 1e-9));
    });
  });

  group('Integration / lifecycle over representative new algorithms', () {
    final makers = <String, TrainerF Function(ANNF)>{
      'Adam': (a) => Adam(a, SamplesSet(xor()), learningRate: 0.05),
      'RMSProp': (a) => RMSProp(a, SamplesSet(xor()), learningRate: 0.02),
      'Lion': (a) => Lion(a, SamplesSet(xor()), learningRate: 0.02),
      'iRProp+': (a) => ResilientPropagation(a, SamplesSet(xor())),
      'LevenbergMarquardt': (a) => LevenbergMarquardt(a, SamplesSet(xor())),
      'GeneticAlgorithm': (a) =>
          GeneticAlgorithm(a, SamplesSet(xor()), random: Random(1)),
    };

    makers.forEach((name, make) {
      test('$name bookkeeping + JSON round-trip + NaN-free', () {
        final t = make(build())..logEnabled = false;

        // Bookkeeping: train() advances epoch/activation counters.
        t.train(20, 0.0);
        expect(t.totalTrainedEpochs, greaterThan(0), reason: '$name epochs');
        expect(t.globalError.isNaN, isFalse, reason: '$name NaN');

        // Persistence: the trained ANN survives a JSON round-trip and can keep
        // training / producing finite outputs.
        final restored = ANN.fromJson(t.ann.toJsonMap()) as ANNF;
        for (final s in xor()) {
          restored.activate(s.input);
          expect(restored.outputAsDouble.first.isFinite, isTrue);
        }
        final err = restored.computeSamplesGlobalError(xor());
        expect(
          err,
          closeTo(t.ann.computeSamplesGlobalError(xor()), 1e-4),
          reason: '$name JSON round-trip error',
        );
      });
    });
  });
}
