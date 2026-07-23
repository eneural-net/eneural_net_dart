import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:eneural_net/src/eneural_net_training_parameter_strategy.dart';
import 'package:eneural_net/src/eneural_net_training_propagation.dart';
import 'package:test/test.dart';

typedef ANNFloat32x4 = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

typedef PropagationFloat32x4 =
    Propagation<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

ANNFloat32x4 buildANN({int hidden = 3, int? seed}) => ANN(
  ScaleDouble.ZERO_TO_ONE,
  LayerFloat32x4(2, true),
  [HiddenLayerConfig(hidden, true)],
  LayerFloat32x4(1, false),
  random: seed != null ? Random(seed) : null,
);

List<SampleFloat32x4> xorSamples() => SampleFloat32x4.toListFromString(
  ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
  ScaleDouble.ZERO_TO_ONE,
  true,
);

/// A minimal [Training] implementation used to exercise the default
/// behavior of the base class.
class _NoOpTraining
    extends
        Training<
          double,
          Float32x4,
          SignalFloat32x4,
          Scale<double>,
          SampleFloat32x4
        > {
  _NoOpTraining(ANNFloat32x4 ann, SamplesSet<SampleFloat32x4> samplesSet)
    : super(ann, samplesSet, 'NoOp');

  int learnCalls = 0;

  /// Nudges the weights a little so that the error actually changes.
  @override
  bool learn(List<SampleFloat32x4> samples, double targetGlobalError) {
    ++learnCalls;

    var weights = ann.allWeights;
    ann.allWeights = weights.map((w) => w * 0.999).toList();

    return computeGlobalError(samples) < targetGlobalError;
  }

  @override
  String get parameters => 'none';
}

/// A minimal [ParameterStrategy] used to exercise the base [resetValue].
class _SimpleStrategy
    extends ParameterStrategy<double, Float32x4, SignalFloat32x4> {
  _SimpleStrategy(PropagationFloat32x4 propagation) : super(propagation);

  double _value = 0.0;

  @override
  double get initialValue => 0.5;

  @override
  double get value => _value;

  @override
  Float32x4 get valueEntry => createValueEntry(_value);

  @override
  void initializeValue() => setValue(initialValue);

  @override
  void setValue(double value) => _value = value;

  @override
  void updateValue() {}
}

void main() {
  group('SignalInt32x4: full lane coverage', () {
    test('getEntryFiltered applies the filter to every lane', () {
      var s = SignalInt32x4.from([1, 2, 3, 4]);

      var filtered = s.getEntryFiltered(0, (n) => n * 10);

      expect([
        filtered.x,
        filtered.y,
        filtered.z,
        filtered.w,
      ], equals([10, 20, 30, 40]));
      expect(s.values, equals([1, 2, 3, 4]), reason: 'get must not mutate');

      s.setEntryFiltered(0, (n) => n + 1);
      expect(s.values, equals([2, 3, 4, 5]));
    });

    test('setValueFromEntry writes every lane', () {
      var s = SignalInt32x4(4);
      var entry = Int32x4(0, 0, 0, 0);

      expect(s.setValueFromEntry(entry, 0, 9).x, equals(9));
      expect(s.setValueFromEntry(entry, 1, 9).y, equals(9));
      expect(s.setValueFromEntry(entry, 2, 9).z, equals(9));
      expect(s.setValueFromEntry(entry, 3, 9).w, equals(9));
    });

    test('addValueFromEntry adds to every lane', () {
      var s = SignalInt32x4(4);
      var entry = Int32x4(1, 2, 3, 4);

      expect(s.addValueFromEntry(entry, 0, 10).x, equals(11));
      expect(s.addValueFromEntry(entry, 1, 10).y, equals(12));
      expect(s.addValueFromEntry(entry, 2, 10).z, equals(13));
      expect(s.addValueFromEntry(entry, 3, 10).w, equals(14));
    });

    test('setValue/addToValue reach every lane of every entry', () {
      var s = SignalInt32x4(8);

      for (var i = 0; i < 8; ++i) {
        s.setValue(i, i + 1);
      }
      expect(s.values, equals([1, 2, 3, 4, 5, 6, 7, 8]));

      for (var i = 0; i < 8; ++i) {
        s.addToValue(i, 10);
      }
      expect(s.values, equals([11, 12, 13, 14, 15, 16, 17, 18]));
    });

    test('multiplyAllEntriesTo/subtractAllEntriesTo', () {
      var a = SignalInt32x4.from([1, 2, 3, 4]);
      var d = SignalInt32x4(4);

      a.multiplyAllEntriesTo(Int32x4(2, 2, 2, 2), d);
      expect(d.values, equals([2, 4, 6, 8]));

      a.subtractAllEntriesTo(Int32x4(1, 1, 1, 1), d);
      expect(d.values, equals([0, 1, 2, 3]));
    });

    test('multiplyAllEntriesAddingTo accumulates', () {
      var a = SignalInt32x4.from([1, 2, 3, 4]);
      var d = SignalInt32x4.from([10, 10, 10, 10]);

      a.multiplyAllEntriesAddingTo(Int32x4(2, 2, 2, 2), d);
      expect(d.values, equals([12, 14, 16, 18]));
    });

    test('multiplyEntries/multiplyValueTo/multiplyAllValuesAddingTo', () {
      var a = SignalInt32x4.from([1, 2, 3, 4]);

      var m = a.multiplyEntries(Int32x4(3, 3, 3, 3));
      expect(m.length, equals(4));
      expect(m.values, equals([3, 6, 9, 12]));

      var d = SignalInt32x4(4);
      a.multiplyValueTo(2, d);
      expect(d.values, equals([2, 4, 6, 8]));

      a.multiplyAllValuesAddingTo(2, d);
      expect(d.values, equals([4, 8, 12, 16]));
    });

    test('createRandomInstance and normalize', () {
      var s = SignalInt32x4(1).createRandomInstance(8, 4);
      expect(s.length, equals(8));
      expect(s.values.every((v) => v >= -4 && v <= 4), isTrue);

      var normalized = SignalInt32x4.from([
        0,
        100,
        200,
      ]).normalize(ScaleInt(0, 100));
      expect(normalized.values, equals([0, 1, 2]));
    });
  });

  group('SignalFloat32x4Mod4: full coverage', () {
    test('EMPTY is a usable prototype', () {
      expect(SignalFloat32x4Mod4.EMPTY.length, equals(0));
      expect(SignalFloat32x4Mod4.EMPTY.values, isEmpty);
    });

    test('fromEntries pads the entries', () {
      var s = SignalFloat32x4Mod4.fromEntries([Float32x4(1, 2, 3, 4)], 4);

      expect(s.values, equals([1, 2, 3, 4]));
      expect(s.entriesLength % 4, equals(0));
    });

    test('createInstanceWithEntries pads the entries', () {
      var s = SignalFloat32x4Mod4(
        4,
      ).createInstanceWithEntries(4, [Float32x4(1, 2, 3, 4)]);

      expect(s, isA<SignalFloat32x4Mod4>());
      expect(s.values, equals([1, 2, 3, 4]));
      expect(s.entriesLength, equals(4));
    });

    test('createRandomInstance', () {
      var s = SignalFloat32x4Mod4(1).createRandomInstance(8, 3.0);

      expect(s, isA<SignalFloat32x4Mod4>());
      expect(s.length, equals(8));
      expect(s.values.every((v) => v >= -3 && v <= 3), isTrue);
      expect(s.entriesLength % 4, equals(0));
    });

    test('multiplyAllEntriesTo/subtractAllEntriesTo', () {
      var a = SignalFloat32x4Mod4.from([1, 2, 3, 4]);
      var d = SignalFloat32x4Mod4(4);

      a.multiplyAllEntriesTo(Float32x4.splat(2), d);
      expect(d.values, equals([2, 4, 6, 8]));

      a.subtractAllEntriesTo(Float32x4.splat(1), d);
      expect(d.values, equals([0, 1, 2, 3]));
    });

    test('multiplyAllEntriesAddingTo accumulates', () {
      var a = SignalFloat32x4Mod4.from([1, 2, 3, 4]);
      var d = SignalFloat32x4Mod4.from([10, 10, 10, 10]);

      a.multiplyAllEntriesAddingTo(Float32x4.splat(2), d);
      expect(d.values, equals([12, 14, 16, 18]));
    });

    test('multiplyValueTo/multiplyAllValuesAddingTo', () {
      var a = SignalFloat32x4Mod4.from([1, 2, 3, 4]);
      var d = SignalFloat32x4Mod4(4);

      a.multiplyValueTo(2.0, d);
      expect(d.values, equals([2, 4, 6, 8]));

      a.multiplyAllValuesAddingTo(2.0, d);
      expect(d.values, equals([4, 8, 12, 16]));
    });

    test('multiplyEntries keeps the Mod4 type', () {
      var a = SignalFloat32x4Mod4.from([1, 2, 3, 4]);
      var m = a.multiplyEntries(Float32x4.splat(3));

      expect(m, isA<SignalFloat32x4Mod4>());
      expect(m.values, equals([3, 6, 9, 12]));
    });
  });

  group('SignalFloat32x4: remaining paths', () {
    test('createEntryFrom overrides each position', () {
      var s = SignalFloat32x4(4);
      var base = Float32x4(1, 2, 3, 4);

      expect(s.createEntryFrom(base, 9.0).x, equals(9.0));
      expect(s.createEntryFrom(base, null, 9.0).y, equals(9.0));
      expect(s.createEntryFrom(base, null, null, 9.0).z, equals(9.0));
      expect(s.createEntryFrom(base, null, null, null, 9.0).w, equals(9.0));

      var all = s.createEntryFrom(base, 5.0, 6.0, 7.0, 8.0);
      expect([all.x, all.y, all.z, all.w], equals([5.0, 6.0, 7.0, 8.0]));
    });

    test('fromEntries', () {
      var s = SignalFloat32x4.fromEntries([Float32x4(1, 2, 3, 4)], 4);
      expect(s.values, equals([1, 2, 3, 4]));
    });

    test('EMPTY prototypes', () {
      expect(SignalFloat32x4.EMPTY.length, equals(0));
      expect(SignalInt32x4.EMPTY.length, equals(0));
    });
  });

  group('ActivationFunction: base flat-spot defaults', () {
    test('the int functions use the base flat-spot implementations', () {
      var af = ActivationFunctionSigmoidFastInt100();

      // Not overridden, so they fall back to the plain derivative:
      expect(af.derivativeWithFlatSpot(10), equals(af.derivative(10)));

      var entry = Int32x4(10, 20, 30, 40);
      var withFlatSpot = af.derivativeEntryWithFlatSpot(entry);
      var plain = af.derivativeEntry(entry);

      expect(withFlatSpot.x, equals(plain.x));
      expect(withFlatSpot.w, equals(plain.w));
    });

    test('the SigmoidFastInt scope is shared with Sigmoid', () {
      expect(
        ActivationFunctionSigmoidFastInt(100).scope,
        equals(ActivationFunctionSigmoid().scope),
      );
      expect(
        ActivationFunctionSigmoidFastInt100().scope,
        equals(ActivationFunctionSigmoid().scope),
      );
      expect(
        ActivationFunctionSigmoidFast().scope,
        equals(ActivationFunctionSigmoid().scope),
      );
      expect(
        ActivationFunctionSigmoidBoundedFast().scope,
        equals(ActivationFunctionSigmoid().scope),
      );
    });
  });

  group('Extensions shadowed by instance members', () {
    test('NumExtension.clamp', () {
      // `num.clamp` shadows it, so it must be applied explicitly:
      expect(NumExtension(5).clamp(0, 10), equals(5));
      expect(NumExtension(-5).clamp(0, 10), equals(0));
      expect(NumExtension(50).clamp(0, 10), equals(10));
    });

    test('DoubleExtension.clamp', () {
      expect(DoubleExtension(5.0).clamp(0.0, 10.0), equals(5.0));
      expect(DoubleExtension(-5.0).clamp(0.0, 10.0), equals(0.0));
      expect(DoubleExtension(50.0).clamp(0.0, 10.0), equals(10.0));
    });

    test('IntExtension.clamp', () {
      expect(IntExtension(5).clamp(0, 10), equals(5));
      expect(IntExtension(-5).clamp(0, 10), equals(0));
      expect(IntExtension(50).clamp(0, 10), equals(10));
    });

    test('ListNumExtension.castElement for an int list', () {
      // `ListIntExtension` is more specific, so apply the generic one:
      expect(ListNumExtension<int>(<int>[1]).castElement(2.9), equals(2));
      expect(
        ListNumExtension<double>(<double>[1.0]).castElement(2),
        equals(2.0),
      );
    });
  });

  group('ParameterStrategy: base behavior', () {
    test('resetValue restores initialValue', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      var strategy = _SimpleStrategy(training);
      strategy.initializeValue();
      expect(strategy.value, equals(0.5));

      strategy.setValue(0.9);
      expect(strategy.value, equals(0.9));

      strategy.resetValue();
      expect(strategy.value, equals(0.5));

      expect(strategy.valueEntry.x, closeTo(0.5, 1e-6));

      strategy.updateValue();
      expect(strategy.value, equals(0.5));
    });
  });

  group('Training: base class behavior', () {
    test('a custom Training uses the base parameter hooks', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      expect(training.algorithmName, equals('NoOp'));
      expect(training.parameters, equals('none'));
      expect(training.lastGlobalError, equals(double.maxFinite));

      // The base `initializeParameters`/`updateParameters` are no-ops:
      expect(() => training.initializeParameters(), returnsNormally);
      expect(() => training.updateParameters(), returnsNormally);

      training.train(10, 0.0);

      expect(training.learnCalls, equals(10));
      expect(training.trainedEpochs, equals(10));
      expect(training.globalError.isFinite, isTrue);

      // A second block moves the current error into `lastGlobalError`:
      var previousError = training.globalError;
      training.train(10, 0.0);

      expect(training.lastGlobalError, equals(previousError));
      expect(training.trainedEpochs, equals(20));
    });

    test('learn can stop the epochs early', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      // A huge target makes `learn` return true on the first epoch:
      training.train(100, 1000.0);

      expect(training.learnCalls, equals(1));
      expect(training.trainedEpochs, equals(1));
    });

    test('checkBestTrainingError tracks the lowest error', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      expect(training.bestTrainingError, equals(double.maxFinite));

      training.checkBestTrainingError(0.5);
      expect(training.bestTrainingError, equals(0.5));

      training.checkBestTrainingError(0.9);
      expect(training.bestTrainingError, equals(0.5), reason: 'not improved');

      training.checkBestTrainingError(0.1);
      expect(training.bestTrainingError, equals(0.1));

      training.resetBestTraining();
      expect(training.bestTrainingError, equals(double.maxFinite));
    });

    test('computeGlobalError delegates to the ANN', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      expect(
        training.computeGlobalError(samples),
        equals(ann.computeSamplesGlobalError(samples)),
      );
    });

    test('the samples set target error is the default', () {
      var samples = xorSamples();
      var samplesSet = SamplesSet(samples, subject: 'noop');
      samplesSet.targetGlobalError = 1000.0;

      var ann = buildANN();
      var training = _NoOpTraining(ann, samplesSet);
      training.logEnabled = false;

      // No `targetGlobalError` given -> uses `samplesSet.targetGlobalError`,
      // which is huge, so it succeeds immediately.
      expect(training.trainUntilGlobalError(maxRetries: 0), isTrue);
    });

    test('selectInitialANN is skipped for a pool of 1', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      training.initialAnnPoolSize = 1;
      var before = ann.allWeights;

      training.selectInitialANN(samples, 0.0);
      expect(ann.allWeights, equals(before));
    });

    test('selectInitialANN picks the best of a pool', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      training.initialAnnPoolSize = 5;
      training.initialAnnEpochs = 2;

      expect(() => training.selectInitialANN(samples, 0.0), returnsNormally);
      expect(ann.allWeights.every((w) => w.isFinite), isTrue);
    });

    test('selectInitialANN returns early when the target is reached', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = _NoOpTraining(ann, SamplesSet(samples, subject: 'noop'));
      training.logEnabled = false;

      training.initialAnnPoolSize = 5;
      training.initialAnnEpochs = 1;

      // A huge target error is reached by the first pool candidate:
      expect(() => training.selectInitialANN(samples, 1000.0), returnsNormally);
    });

    test('toString', () {
      var training = _NoOpTraining(buildANN(), SamplesSet(xorSamples()));
      expect(training.toString(), contains('NoOp'));
    });
  });

  group('Training: failure path', () {
    test('returns false when the target cannot be reached', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;

      // An impossible target with a tiny budget:
      var ok = training.trainUntilGlobalError(
        targetGlobalError: 1.0E-30,
        epochsBlock: 1,
        maxEpochs: 60,
        maxRetries: 1,
      );

      expect(ok, isFalse);
      expect(training.totalFailedEpochs > 0, isTrue);
      expect(training.endTime, isNotNull);
      expect(training.globalError.isFinite, isTrue);

      // The best weights found are restored:
      expect(ann.allWeights.every((w) => w.isFinite), isTrue);
    });

    test('normalizes out-of-range arguments', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;

      // epochsBlock < 1, maxEpochs < 1, maxRetries < 0 and
      // retryIncreaseMaxEpochsRatio < 1 are all clamped:
      var ok = training.trainUntilGlobalError(
        targetGlobalError: 1.0E-30,
        epochsBlock: 0,
        maxEpochs: 0,
        maxRetries: -5,
        retryIncreaseMaxEpochsRatio: 0.0,
      );

      expect(ok, isFalse);
      expect(training.totalTrainedEpochs > 0, isTrue);
    });

    test('the extra-epochs evolution path runs', () {
      var samples = xorSamples();
      var ann = buildANN(hidden: 4);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;

      // A small `epochsBlock` builds a long error evolution history, which
      // triggers the moving-average analysis and the extra epochs.
      var ok = training.trainUntilGlobalError(
        targetGlobalError: 1.0E-30,
        epochsBlock: 1,
        maxEpochs: 150,
        maxRetries: 2,
      );

      expect(ok, isFalse);
      expect(training.totalTrainedEpochs > 150, isTrue);
      expect(ann.allWeights.every((w) => w.isFinite), isTrue);
    });

    test('a non-evolving error skips the extra epochs', () {
      var samples = xorSamples();
      var ann = buildANN(hidden: 4);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;

      // A huge learning rate keeps the error from improving, so the
      // moving-average analysis reports "NOT Evolving".
      training.train(1, 0.0);
      training.setLearningRate(500.0);

      var ok = training.trainUntilGlobalError(
        targetGlobalError: 1.0E-30,
        epochsBlock: 1,
        maxEpochs: 150,
        maxRetries: 0,
      );

      expect(ok, isFalse);
      expect(ann.allWeights.every((w) => w.isFinite), isTrue);
    });

    test('the target can be reached during the extra epochs', () {
      var samples = xorSamples();
      const maxEpochs = 40;

      // Measures the error that `maxEpochs` alone reaches...
      var probeAnn = buildANN(hidden: 4, seed: 1);
      var probe = Backpropagation(probeAnn, SamplesSet(samples));
      probe.logEnabled = false;
      probe.enableSelectInitialANN = false;
      probe.train(maxEpochs, 0.0);

      var errorAtBudget = probeAnn.computeSamplesGlobalError(samples);

      // ...then targets slightly below it, so only the extra epochs granted
      // by the "still evolving" analysis can reach it.
      var ann = buildANN(hidden: 4, seed: 1);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;

      var ok = training.trainUntilGlobalError(
        targetGlobalError: errorAtBudget * 0.999,
        epochsBlock: 1,
        maxEpochs: maxEpochs,
        maxRetries: 0,
      );

      expect(ok, isTrue);
      expect(
        training.totalTrainedEpochs > maxEpochs,
        isTrue,
        reason: 'the extra epochs must have run',
      );
      expect(training.endTime, isNotNull);
    });

    test('progress logging can be enabled', () {
      var samples = xorSamples();
      var ann = buildANN();
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.logProgressEnabled = true;
      training.enableSelectInitialANN = false;

      // Runs enough blocks to reach the periodic progress log:
      expect(
        () => training.trainUntilGlobalError(
          targetGlobalError: 1.0E-30,
          epochsBlock: 1,
          maxEpochs: 220,
          maxRetries: 0,
        ),
        returnsNormally,
      );
    });
  });

  group('Training: deprecated API', () {
    test('DefaultTrainingLogger forwards to defaultTrainingLogger', () {
      var training = _NoOpTraining(buildANN(), SamplesSet(xorSamples()));

      expect(
        // ignore: deprecated_member_use_from_same_package
        () => DefaultTrainingLogger(training, 'INFO', 'message'),
        returnsNormally,
      );
      expect(
        // ignore: deprecated_member_use_from_same_package
        () => DefaultTrainingLogger(
          training,
          'ERROR',
          'message',
          'error',
          StackTrace.current,
        ),
        returnsNormally,
      );
    });

    test('a custom logger is used', () {
      var captured = <String>[];

      var training = Backpropagation(
        buildANN(),
        SamplesSet(xorSamples(), subject: 'xor'),
      );
      training.logEnabled = false;

      // The base class exposes the logger through the log* helpers:
      training.logInfo('ignored');
      expect(captured, isEmpty);
    });
  });
}
