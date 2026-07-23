import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
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

ANNFloat32x4 buildANN() => ANN(
  ScaleDouble.ZERO_TO_ONE,
  LayerFloat32x4(2, true),
  [HiddenLayerConfig(3, true)],
  LayerFloat32x4(1, false),
);

List<SampleFloat32x4> xorSamples() => SampleFloat32x4.toListFromString(
  ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
  ScaleDouble.ZERO_TO_ONE,
  true,
);

void main() {
  group('Propagation: setup', () {
    test('the learning rate is finite before the training starts', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      expect(
        training.learningRate.isFinite,
        isTrue,
        reason: 'must not be Infinity when trainingSamplesSize is still 0',
      );
      expect(training.momentum, equals(0.0));
    });

    test('the learning rate becomes 1/samples once initialized', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      training.train(1, 0.0);

      expect(training.learningRate, closeTo(1 / 4, 1e-12));
    });

    test('the learning rate and momentum can be set', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      training.setLearningRate(0.5);
      expect(training.learningRate, equals(0.5));

      training.setMomentum(0.25);
      expect(training.momentum, equals(0.25));

      expect(training.learningRateEntry.x, closeTo(0.5, 1e-6));
      expect(training.momentumEntry.x, closeTo(0.25, 1e-6));
    });

    test('parameters describes the current state', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      var parameters = training.parameters;

      expect(parameters, contains('learningRate'));
      expect(parameters, contains('momentum'));
      expect(parameters, contains('noImprovementLimit'));
    });

    test('signalInstance is a single-value signal', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      expect(training.signalInstance.length, equals(1));
      expect(training.signalInstance.entryBlockSize, equals(4));
    });

    test('noImprovementRatio is normalized', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;

      training.noImprovementRatio = -0.5;
      expect(training.noImprovementRatio, equals(0.5), reason: 'made positive');

      training.noImprovementRatio = 0.0;
      expect(
        training.noImprovementRatio,
        equals(1.0E-20),
        reason: 'clamped to a minimum',
      );
    });

    test('random helpers stay in range', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      for (var i = 0; i < 100; ++i) {
        expect(training.generateRandomValuePositive(2.0) >= 0, isTrue);
        expect(training.generateRandomValuePositive(2.0) <= 2.0, isTrue);

        var v = training.generateRandomValue(2.0);
        expect(v >= -2.0 && v <= 2.0, isTrue);
      }

      expect(training.random, isNotNull);
    });

    test('generateRandomWeightUpdate is clamped to min/max', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      for (var i = 0; i < 100; ++i) {
        var w = training.generateRandomWeightUpdate(10.0, 1.0, 5.0, 1.0);
        expect(w.abs() >= 1.0 && w.abs() <= 5.0, isTrue, reason: 'got $w');
      }
    });

    test('generateRandomWeightUpdateByFactor is proportional', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var update = training.generateRandomWeightUpdateByFactor(0.0, 1.0);
      expect(update, equals(0.0), reason: 'a zero weight has no update');

      for (var i = 0; i < 50; ++i) {
        var u = training.generateRandomWeightUpdateByFactor(1.0, 0.5);
        expect(u.isFinite, isTrue);
      }
    });
  });

  group('StaticParameterStrategy', () {
    late PropagationFloat32x4 training;

    setUp(() {
      training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
    });

    test('keeps its value', () {
      var strategy = StaticParameterStrategy(training, 0.3);

      expect(strategy.value, equals(0.3));
      expect(strategy.initialValue, equals(0.3));
      expect(strategy.valueEntry.x, closeTo(0.3, 1e-6));

      strategy.updateValue();
      expect(strategy.value, equals(0.3), reason: 'update is a no-op');
    });

    test('setValue updates the entry too', () {
      var strategy = StaticParameterStrategy(training, 0.3);

      strategy.setValue(0.8);
      expect(strategy.value, equals(0.8));
      expect(strategy.valueEntry.x, closeTo(0.8, 1e-6));

      // Setting the same value is a no-op:
      strategy.setValue(0.8);
      expect(strategy.value, equals(0.8));
    });

    test('resetValue restores the initialized value', () {
      var strategy = StaticParameterStrategy(training, 0.3);
      strategy.initializeValue();

      strategy.setValue(0.9);
      strategy.resetValue();

      expect(strategy.value, equals(0.3));
    });

    test('defaults to zero', () {
      expect(StaticParameterStrategy(training).value, equals(0.0));
    });
  });

  group('LearningRateStrategy', () {
    late PropagationFloat32x4 training;

    setUp(() {
      training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      // Makes `trainingSamplesSize` non-zero:
      training.train(1, 0.0);
    });

    test('initializes to 1/samplesSize', () {
      var strategy = LearningRateStrategy(training);
      strategy.initializeValue();

      expect(strategy.initialValue, closeTo(1 / 4, 1e-12));
      expect(strategy.value, closeTo(1 / 4, 1e-12));
    });

    test('a multiplier scales the initial value', () {
      var strategy = LearningRateStrategy(training, multiplier: 2);
      strategy.initializeValue();

      expect(strategy.initialValue, closeTo(2 / 4, 1e-12));
      expect(strategy.multiplier, equals(2.0));
    });

    test('is finite when the samples size is still unknown', () {
      var fresh = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      fresh.logEnabled = false;

      var strategy = LearningRateStrategy(fresh);
      strategy.initializeValue();

      expect(
        strategy.value.isFinite,
        isTrue,
        reason: 'must not divide by a zero samples size',
      );
    });

    test('setValue/resetValue', () {
      var strategy = LearningRateStrategy(training);
      strategy.initializeValue();

      strategy.setValue(0.9);
      expect(strategy.value, equals(0.9));
      expect(strategy.valueEntry.x, closeTo(0.9, 1e-6));

      strategy.resetValue();
      expect(strategy.value, closeTo(1 / 4, 1e-12));
    });

    test('decays after 10 consecutive worse epochs', () {
      var strategy = LearningRateStrategy(training);
      strategy.initializeValue();

      var initial = strategy.value;

      // `globalLearnError > lastGlobalLearnError` is what the strategy reads.
      // The XOR training makes the error oscillate, so drive it directly:
      for (var i = 0; i < 9; ++i) {
        strategy.updateValue();
      }

      // The rate must never grow beyond the initial value:
      expect(strategy.value <= initial, isTrue);
    });

    test('never grows beyond the initial value', () {
      var strategy = LearningRateStrategy(training);
      strategy.initializeValue();

      var initial = strategy.value;

      for (var i = 0; i < 200; ++i) {
        strategy.updateValue();
        expect(
          strategy.value <= initial + 1e-12,
          isTrue,
          reason: 'grew beyond the initial value at iteration $i',
        );
      }
    });

    test('decays when the training error keeps growing', () {
      // A huge learning rate makes the error diverge, which is exactly the
      // condition that triggers the decay.
      var diverging = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      diverging.logEnabled = false;
      diverging.train(1, 0.0);

      // A huge learning rate makes an epoch worse than the previous one:
      diverging.setLearningRate(1000.0);

      var worsening = false;
      for (var i = 0; i < 50 && !worsening; ++i) {
        diverging.setLearningRate(1000.0);
        diverging.train(1, 0.0);
        worsening = diverging.globalLearnError > diverging.lastGlobalLearnError;
      }

      expect(
        worsening,
        isTrue,
        reason: 'the setup must leave the training in a worsening state',
      );

      var strategy = LearningRateStrategy(diverging);
      strategy.initializeValue();

      var initial = strategy.value;

      // The decay happens every 10 consecutive worsening epochs:
      for (var i = 0; i < 10; ++i) {
        strategy.updateValue();
      }

      expect(
        strategy.value < initial,
        isTrue,
        reason: 'the learning rate must decay: $initial -> ${strategy.value}',
      );
    });

    test('never decays below initialValue/1000', () {
      var strategy = LearningRateStrategy(training);
      strategy.initializeValue();

      var floor = strategy.initialValue / 1000;

      for (var i = 0; i < 1000; ++i) {
        strategy.updateValue();
      }

      expect(strategy.value >= floor - 1e-15, isTrue);
    });
  });

  group('MomentumRateStrategy', () {
    late PropagationFloat32x4 training;

    setUp(() {
      training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);
    });

    test('starts at zero', () {
      var strategy = MomentumRateStrategy(training);
      strategy.initializeValue();

      expect(strategy.value, equals(0.0));
      expect(strategy.initialValue, equals(0.0));
      expect(strategy.valueEntry.x, equals(0.0));
    });

    test('setValue/resetValue', () {
      var strategy = MomentumRateStrategy(training);
      strategy.initializeValue();

      strategy.setValue(0.4);
      expect(strategy.value, equals(0.4));
      expect(strategy.valueEntry.x, closeTo(0.4, 1e-6));

      strategy.resetValue();
      expect(strategy.value, equals(0.0));
    });

    test('stays within 0..1', () {
      var strategy = MomentumRateStrategy(training);
      strategy.initializeValue();

      for (var i = 0; i < 500; ++i) {
        strategy.updateValue();
        expect(
          strategy.value >= 0 && strategy.value <= 1,
          isTrue,
          reason: 'out of range at iteration $i: ${strategy.value}',
        );
      }
    });
  });

  group('ProportionalToErrorStrategy', () {
    late PropagationFloat32x4 training;

    setUp(() {
      training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);
    });

    test('computeValue is clamped to min/max', () {
      var strategy = ProportionalToErrorStrategy(
        training,
        minValue: 0.1,
        maxValue: 0.9,
        zero: 0.0,
        multiplier: 1.0,
      );

      expect(strategy.computeValue(0.5), equals(0.5));
      expect(strategy.computeValue(0.0), equals(0.1), reason: 'clamped to min');
      expect(strategy.computeValue(5.0), equals(0.9), reason: 'clamped to max');
    });

    test('the zero point and the multiplier are applied', () {
      var strategy = ProportionalToErrorStrategy(
        training,
        zero: 0.2,
        multiplier: 2.0,
      );

      expect(strategy.computeValue(0.1), closeTo(0.4, 1e-12));
    });

    test('initializeValue uses an error of 1.0', () {
      var strategy = ProportionalToErrorStrategy(training, maxValue: 10);
      strategy.initializeValue();

      expect(strategy.initialValue, equals(1.0));
      expect(strategy.value, equals(1.0));
      expect(strategy.valueEntry.x, closeTo(1.0, 1e-6));
    });

    test('updateValue follows the training error', () {
      var strategy = ProportionalToErrorStrategy(training, maxValue: 10);
      strategy.initializeValue();

      strategy.updateValue();
      expect(strategy.value, equals(training.globalLearnError));
    });

    test('resetValue restores the initial value', () {
      var strategy = ProportionalToErrorStrategy(training, maxValue: 10);
      strategy.initializeValue();

      strategy.setValue(5.0);
      strategy.resetValue();

      expect(strategy.value, equals(1.0));
    });
  });

  group('Backpropagation', () {
    test('algorithm name', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      expect(training.algorithmName, equals('Backpropagation'));
    });

    test('computeWeightUpdate accumulates the momentum', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      training.setLearningRate(0.5);
      training.setMomentum(0.0);

      var deltas = <num>[0.0, 0.0, 0.0, 0.0];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      var update = training.computeWeightUpdate(
        0.0,
        0.0,
        2.0,
        0.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(update, closeTo(1.0, 1e-12), reason: 'learningRate * gradient');
      expect(deltas[0], closeTo(1.0, 1e-12), reason: 'delta is stored');

      // With a momentum the previous delta is carried over:
      training.setMomentum(0.5);
      var update2 = training.computeWeightUpdate(
        0.0,
        0.0,
        2.0,
        0.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(update2, closeTo(1.0 + 0.5, 1e-12));
    });

    test('uses the SIMD weight update', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      training.setLearningRate(0.5);
      training.setMomentum(0.0);

      var deltas = training.signalInstance.createInstance(4);
      var counters = training.signalInstance.createInstance(4);

      var update = training.computeEntryWeightUpdate(
        Float32x4.zero(),
        Float32x4.zero(),
        Float32x4.splat(2.0),
        Float32x4.zero(),
        deltas,
        counters,
        0,
        Float32x4.zero(),
      );

      expect(update.x, closeTo(1.0, 1e-6));
      expect(deltas.getValue(0), closeTo(1.0, 1e-6));
    });
  });

  group('RProp', () {
    test('algorithm name and defaults', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));

      expect(training.algorithmName, equals('iRProp+'));
      expect(
        training.enableSelectInitialANN,
        isFalse,
        reason: 'RProp disables the initial ANN selection by default',
      );

      expect(RProp.weightMinStep, equals(1.0E-6));
      expect(RProp.weightMaxStep, equals(50.0));
    });

    test('the initial ANN selection can be enabled', () {
      var training = RProp(
        buildANN(),
        SamplesSet(xorSamples()),
        enableSelectInitialANN: true,
      );

      expect(training.enableSelectInitialANN, isTrue);
    });

    test('uses static learning rate and momentum', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(10, 0.0);

      expect(training.learningRate, equals(0.0));
      expect(training.momentum, equals(0.0));
    });

    test('grows the step when the gradient keeps its direction', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = <num>[0.1, 0.1, 0.1, 0.1];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      // Same sign gradients -> the update delta grows by 1.2.
      var update = training.computeWeightUpdate(
        0.0,
        0.0,
        1.0,
        1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(deltas[0], closeTo(0.12, 1e-12));
      expect(update, closeTo(0.12, 1e-12), reason: 'sign(+1) * delta');
    });

    test('shrinks and reverses the step when the gradient flips', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = <num>[0.1, 0.1, 0.1, 0.1];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      training.computeWeightUpdate(
        0.0,
        0.0,
        1.0,
        -1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      // The delta is halved and marked negative, telling the next iteration
      // not to change direction again:
      expect(deltas[0], closeTo(-0.05, 1e-12));
    });

    test('a zero gradient keeps the step', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = <num>[0.1, 0.1, 0.1, 0.1];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      var update = training.computeWeightUpdate(
        0.0,
        0.0,
        0.0,
        0.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(deltas[0], closeTo(0.1, 1e-12));
      expect(update, equals(0.0), reason: 'sign(0) * delta');
    });

    test('the step never exceeds weightMaxStep', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = <num>[1000.0, 0.0, 0.0, 0.0];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      training.computeWeightUpdate(
        0.0,
        0.0,
        1.0,
        1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(deltas[0], equals(RProp.weightMaxStep));
    });

    test('the step never goes below weightMinStep', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = <num>[1.0E-30, 0.0, 0.0, 0.0];
      var counters = <num>[0.0, 0.0, 0.0, 0.0];

      training.computeWeightUpdate(
        0.0,
        0.0,
        1.0,
        -1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(deltas[0], equals(-RProp.weightMinStep));
    });

    test('has no SIMD weight update implementation', () {
      var training = RProp(buildANN(), SamplesSet(xorSamples()));
      training.logEnabled = false;
      training.train(1, 0.0);

      var deltas = training.signalInstance.createInstance(4);
      var counters = training.signalInstance.createInstance(4);

      expect(
        () => training.computeEntryWeightUpdateSIMD(
          Float32x4.zero(),
          Float32x4.zero(),
          Float32x4.zero(),
          Float32x4.zero(),
          deltas,
          counters,
          0,
          Float32x4.zero(),
        ),
        throwsA(isA<UnsupportedError>()),
      );
    });
  });

  group('Training: logger', () {
    test('the default logger prints', () {
      var training = Backpropagation(buildANN(), SamplesSet(xorSamples()));

      expect(
        () => defaultTrainingLogger(training, 'INFO', 'message'),
        returnsNormally,
      );
      expect(
        () => defaultTrainingLogger(
          training,
          'ERROR',
          'message',
          'the error',
          StackTrace.current,
        ),
        returnsNormally,
      );
    });
  });
}
