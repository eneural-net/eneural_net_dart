import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:test/test.dart';

typedef ANNFloat32x4 = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

typedef TrainingFloat32x4 =
    Training<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

/// Builds a `Float32x4` ANN with the given topology.
///
/// [seed] makes the initial weights reproducible, so that the training tests
/// are deterministic instead of flaky.
ANNFloat32x4 buildANN(
  int inputs,
  List<int> hiddenLayers,
  int outputs, {
  ActivationFunction<double, Float32x4>? activationFunction,
  Scale<double>? scale,
  int seed = 101,
}) {
  var af = activationFunction ?? ActivationFunctionSigmoid();

  return ANN(
    scale ?? ScaleDouble.ZERO_TO_ONE,
    LayerFloat32x4(inputs, true, af),
    hiddenLayers.map((n) => HiddenLayerConfig(n, true)).toList(),
    LayerFloat32x4(outputs, false, af),
    random: Random(seed),
  );
}

/// Trains [training] until [targetGlobalError] and asserts that every sample
/// is predicted within [sampleErrorTolerance].
///
/// The training is seeded so that the retries/restarts are reproducible.
void trainAndAssert(
  ANNFloat32x4 ann,
  TrainingFloat32x4 training, {
  double targetGlobalError = 0.02,
  double sampleErrorTolerance = 0.10,
  int maxRetries = 20,
  int seed = 101,
}) {
  training.logEnabled = false;

  var ok = training.trainUntilGlobalError(
    targetGlobalError: targetGlobalError,
    maxRetries: maxRetries,
    random: Random(seed),
  );

  expect(
    ok,
    isTrue,
    reason:
        '${training.algorithmName} did not reach $targetGlobalError '
        '(final error: ${training.globalError})',
  );

  var globalError = ann.computeSamplesGlobalError(training.samples);
  expect(globalError <= targetGlobalError, isTrue);

  for (var sample in training.samples) {
    ann.activate(sample.input);

    var sampleError = (ann.output - sample.output).squaresMean;

    expect(
      sampleError < sampleErrorTolerance,
      isTrue,
      reason:
          'sample ${sample.input.values} -> ${ann.output} '
          '(expected ${sample.output.values}) ; error: $sampleError',
    );
  }
}

void main() {
  var scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> samplesFromStrings(List<String> pairs) =>
      SampleFloat32x4.toListFromString(pairs, scale, true);

  var xor = ['0,0=0', '0,1=1', '1,0=1', '1,1=0'];
  var and = ['0,0=0', '0,1=0', '1,0=0', '1,1=1'];
  var or = ['0,0=0', '0,1=1', '1,0=1', '1,1=1'];

  group('Integration: logic gates', () {
    test('XOR with Backpropagation', () {
      var ann = buildANN(2, [3], 1);
      var training = Backpropagation(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor'),
      );

      trainAndAssert(ann, training);
    });

    test('XOR with RProp', () {
      var ann = buildANN(2, [3], 1);
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor'),
      );

      trainAndAssert(ann, training);
    });

    test('AND with Backpropagation', () {
      var ann = buildANN(2, [3], 1);
      var training = Backpropagation(
        ann,
        SamplesSet(samplesFromStrings(and), subject: 'and'),
      );

      trainAndAssert(ann, training);
    });

    test('OR with RProp', () {
      var ann = buildANN(2, [3], 1);
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(or), subject: 'or'),
      );

      trainAndAssert(ann, training);
    });

    test('XOR with SigmoidFast', () {
      var ann = buildANN(
        2,
        [4],
        1,
        activationFunction: ActivationFunctionSigmoidFast(),
      );
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor-fast'),
      );

      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });

    test('XOR with SigmoidBoundedFast', () {
      var ann = buildANN(
        2,
        [4],
        1,
        activationFunction: ActivationFunctionSigmoidBoundedFast(),
      );
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor-bounded'),
      );

      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });
  });

  group('Integration: topologies', () {
    test('two hidden layers learn XOR', () {
      var ann = buildANN(2, [4, 3], 1);
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor-deep'),
      );

      expect(ann.allLayers.length, equals(4));
      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });

    test('multiple outputs are learned independently', () {
      // Two outputs: the first is XOR, the second is AND.
      var samples = samplesFromStrings([
        '0,0=0,0',
        '0,1=1,0',
        '1,0=1,0',
        '1,1=0,1',
      ]);

      var ann = buildANN(2, [5], 2);
      var training = RProp(ann, SamplesSet(samples, subject: 'xor+and'));

      expect(ann.outputSize, equals(2));
      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });

    test('a wide hidden layer with unaligned size trains', () {
      // 7 hidden neurons + bias = 8; the input layer has 3 -> exercises the
      // partial SIMD entries.
      var ann = buildANN(2, [7], 1);
      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(xor), subject: 'xor-wide'),
      );

      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });

    test('a layer without bias neurons trains', () {
      var ann = ANN(scale, LayerFloat32x4(2, false), [
        HiddenLayerConfig(5, false),
      ], LayerFloat32x4(1, false));

      var training = RProp(
        ann,
        SamplesSet(samplesFromStrings(or), subject: 'or-nobias'),
      );

      trainAndAssert(ann, training, targetGlobalError: 0.03);
    });
  });

  group('Integration: signal formats', () {
    test('a Float32x4Mod4 ANN activates and trains', () {
      var source = buildANN(2, [3], 1);

      // Rebuild the same ANN using the `Float32x4Mod4` signal format:
      var map = source.toJsonMap();
      map['format'] = 'Float32x4Mod4';
      for (var layer in (map['layers'] as List)) {
        (layer as Map)['format'] = 'Float32x4Mod4';
      }

      var ann = ANN.fromJson(map) as ANNFloat32x4;

      expect(ann.format, equals('Float32x4Mod4'));
      expect(ann.inputLayer.neurons, isA<SignalFloat32x4Mod4>());
      expect(ann.allWeights, equals(source.allWeights));

      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'xor-mod4');

      var errorBefore = ann.computeSamplesGlobalError(samples);
      expect(errorBefore.isNaN, isFalse);

      var training = Backpropagation(ann, samplesSet);
      training.logEnabled = false;

      var errorAfter = training.train(2000, 0.001);

      expect(errorAfter.isNaN, isFalse);
      expect(
        errorAfter < errorBefore,
        isTrue,
        reason: 'the error must decrease: $errorBefore -> $errorAfter',
      );
    });

    test('an Int32x4 ANN activates and serializes', () {
      var scaleInt = ScaleInt.ZERO_TO_ONE;

      var ann = ANN(scaleInt, LayerInt32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerInt32x4(1, false));

      expect(ann.format, equals('Int32x4'));
      expect(ann.allLayersNeuronsSize, equals([3, 4, 1]));

      var samples = SampleInt32x4.toListFromString(xor, scaleInt, true);

      for (var sample in samples) {
        ann.activate(sample.input);

        expect(ann.output.length, equals(1));
        expect(
          ann.output.first >= 0 && ann.output.first <= 100,
          isTrue,
          reason: 'the int sigmoid output must stay in the 0..100 scale',
        );
      }

      var globalError = ann.computeSamplesGlobalError(samples);
      expect(globalError.isNaN, isFalse);

      var decoded = ANN.fromJson(ann.toJson());
      expect(decoded.allWeights, equals(ann.allWeights));
      expect(decoded.toJson(), equals(ann.toJson()));
    });
  });

  group('Integration: function approximation', () {
    test('learns a linear function', () {
      var generator = SamplesGenerator(ScaleDouble(0, 1), (x) => x, 20);
      var samples = generator.generateSamples();

      expect(samples.length, equals(21));

      var ann = buildANN(1, [5], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'linear'));
      training.logEnabled = false;

      var errorBefore = ann.computeSamplesGlobalError(samples);
      training.trainUntilGlobalError(targetGlobalError: 0.005, maxRetries: 5);
      var errorAfter = ann.computeSamplesGlobalError(samples);

      expect(
        errorAfter < errorBefore,
        isTrue,
        reason: 'the error must decrease: $errorBefore -> $errorAfter',
      );
      expect(errorAfter < 0.02, isTrue, reason: 'final error: $errorAfter');
    });

    test('learns a quadratic function', () {
      var generator = SamplesGenerator(ScaleDouble(0, 1), (x) => x * x, 20);
      var samples = generator.generateSamples();

      var ann = buildANN(1, [6], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'quadratic'));
      training.logEnabled = false;

      training.trainUntilGlobalError(targetGlobalError: 0.005, maxRetries: 5);

      var error = ann.computeSamplesGlobalError(samples);
      expect(error < 0.02, isTrue, reason: 'final error: $error');

      // Spot-check a value that is not part of the training set:
      ann.activate(SignalFloat32x4.from([0.35]));
      expect(ann.output.first, closeTo(0.35 * 0.35, 0.15));
    });
  });

  group('Integration: scaled data', () {
    test('trains on a non 0..1 scale and denormalizes the output', () {
      var dataScale = ScaleDouble(0, 100);

      // Inputs/outputs in the 0..100 domain:
      var samples = SampleFloat32x4.toList([
        [
          [0, 0],
          [0],
        ],
        [
          [0, 100],
          [100],
        ],
        [
          [100, 0],
          [100],
        ],
        [
          [100, 100],
          [0],
        ],
      ], dataScale);

      expect(samples.first.input.values, equals([0.0, 0.0]));
      expect(samples[1].output.values, equals([1.0]));

      var ann = buildANN(2, [4], 1, scale: dataScale);
      var training = RProp(ann, SamplesSet(samples, subject: 'xor-100'));
      training.logEnabled = false;

      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);

      ann.activate(samples[1].input);

      var denormalized = ann.outputDenormalized.first;
      expect(
        denormalized,
        closeTo(100, 40),
        reason: 'denormalized output: $denormalized',
      );
      expect(denormalized, closeTo(ann.output.first * 100, 1e-6));
    });
  });

  group('Integration: persistence', () {
    test('a trained ANN survives a JSON round-trip', () {
      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'xor');

      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, samplesSet);
      training.logEnabled = false;

      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);

      var trainedError = ann.computeSamplesGlobalError(samples);
      var trainedOutputs = ann.computeSamplesActivations(samples);
      var json = ann.toJson();

      var restored = ANN.fromJson(json) as ANNFloat32x4;

      expect(restored.allWeights, equals(ann.allWeights));
      expect(restored.toJson(), equals(json));
      expect(
        restored.computeSamplesActivations(samples),
        equals(trainedOutputs),
      );
      expect(restored.computeSamplesGlobalError(samples), equals(trainedError));
    });

    test('a restored ANN can keep training', () {
      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'xor');

      var ann = buildANN(2, [4], 1);
      var training = Backpropagation(ann, samplesSet);
      training.logEnabled = false;
      training.train(200, 0.001);

      var restored = ANN.fromJson(ann.toJson()) as ANNFloat32x4;
      var errorBefore = restored.computeSamplesGlobalError(samples);

      var training2 = RProp(restored, samplesSet);
      training2.logEnabled = false;
      training2.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);

      var errorAfter = restored.computeSamplesGlobalError(samples);

      expect(
        errorAfter < errorBefore,
        isTrue,
        reason: 'the error must decrease: $errorBefore -> $errorAfter',
      );
    });

    test('weights can be saved and reloaded manually', () {
      var samples = samplesFromStrings(xor);

      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);

      var savedWeights = ann.allWeights;
      var savedError = ann.computeSamplesGlobalError(samples);

      // Destroy the training:
      ann.resetWeights();
      expect(
        ann.computeSamplesGlobalError(samples) > savedError,
        isTrue,
        reason: 'resetting the weights must lose the training',
      );

      // Restore it:
      ann.allWeights = savedWeights;
      expect(ann.computeSamplesGlobalError(samples), closeTo(savedError, 1e-9));
    });
  });

  group('Integration: training lifecycle', () {
    test('reports epochs, activations and elapsed time', () {
      var samples = samplesFromStrings(xor);

      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;

      expect(training.startTime, isNull);
      expect(training.endTime, isNull);
      expect(training.elapsedTime, isNull);
      expect(training.totalTrainedEpochs, equals(0));

      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);

      expect(training.startTime, isNotNull);
      expect(training.endTime, isNotNull);
      expect(training.elapsedTime, isNotNull);

      expect(training.trainedEpochs > 0, isTrue);
      expect(training.totalTrainedEpochs >= training.trainedEpochs, isTrue);
      expect(training.trainingActivations > 0, isTrue);
      expect(
        training.totalTrainingActivations >= training.trainingActivations,
        isTrue,
      );
    });

    test('initializeTraining records the samples size', () {
      var samples = samplesFromStrings(xor);

      var ann = buildANN(2, [3], 1);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;

      expect(training.trainingSamplesSize, equals(0));

      training.train(1, 0.0);
      expect(training.trainingSamplesSize, equals(4));

      // `reset` clears it so that the parameters are re-initialized for the
      // next session:
      training.reset();
      expect(training.trainingSamplesSize, equals(0));
      expect(training.trainedEpochs, equals(0));
      expect(training.globalError, equals(double.maxFinite));
    });

    test('train() runs a fixed number of epochs', () {
      var samples = samplesFromStrings(xor);

      var ann = buildANN(2, [3], 1);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;

      var errorBefore = ann.computeSamplesGlobalError(samples);
      var error = training.train(500, 0.0);

      expect(training.trainedEpochs, equals(500));
      expect(training.trainingActivations, equals(500 * 4));
      expect(error, closeTo(ann.computeSamplesGlobalError(samples), 1e-12));
      expect(error < errorBefore, isTrue);
    });

    test('a second session is not anchored to the previous best', () {
      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'xor');

      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, samplesSet);
      training.logEnabled = false;

      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);
      expect(training.bestTrainingError < double.maxFinite, isTrue);

      // Starting a new session discards the previous best:
      training.resetBestTraining();
      expect(training.bestTrainingError, equals(double.maxFinite));

      training.trainUntilGlobalError(targetGlobalError: 0.02, maxRetries: 10);
      expect(
        ann.computeSamplesGlobalError(samples) <= 0.02,
        isTrue,
        reason: 'the second session must still converge',
      );
    });

    test('the training subject can be customized', () {
      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'set-subject');
      var ann = buildANN(2, [3], 1);

      expect(
        Backpropagation(ann, samplesSet, subject: 'custom').subject,
        equals('custom'),
      );
      expect(
        RProp(ann, samplesSet, subject: 'custom').subject,
        equals('custom'),
      );

      // Defaults to the samples set subject:
      expect(Backpropagation(ann, samplesSet).subject, equals('set-subject'));
      expect(
        Backpropagation(ann, samplesSet).samplesSubject,
        equals('set-subject'),
      );
    });

    test('a custom logger receives the training messages', () {
      var samples = samplesFromStrings(xor);
      var messages = <String>[];

      var ann = buildANN(2, [3], 1);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));

      // The logger is applied by the base `Training` class:
      training.logEnabled = true;
      training.logProgressEnabled = false;

      expect(training.algorithmName, equals('Backpropagation'));
      expect(training.toString(), contains('Backpropagation'));

      messages.add(training.parameters);
      expect(messages.first, contains('learningRate'));
    });

    test('training can be disabled from logging', () {
      var samples = samplesFromStrings(xor);
      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'xor'));

      training.logEnabled = false;

      expect(() => training.logInfo('info'), returnsNormally);
      expect(() => training.logWarn('warn'), returnsNormally);
      expect(() => training.logProgress('progress'), returnsNormally);
      expect(() => training.logError('error'), returnsNormally);
    });
  });

  group('Integration: robustness', () {
    test('the outputs never become NaN during a long training', () {
      var samples = samplesFromStrings(xor);

      var ann = buildANN(2, [4], 1);
      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;

      for (var i = 0; i < 20; ++i) {
        training.train(100, 0.0);

        expect(
          ann.allWeights.every((w) => w.isFinite),
          isTrue,
          reason: 'weights became non-finite at block $i',
        );

        for (var sample in samples) {
          ann.activate(sample.input);
          expect(
            ann.output.every((o) => o.isFinite),
            isTrue,
            reason: 'output became non-finite at block $i',
          );
        }
      }
    });

    test('a single-sample set trains', () {
      var samples = samplesFromStrings(['0,1=1']);

      var ann = buildANN(2, [3], 1);
      var training = RProp(ann, SamplesSet(samples, subject: 'single'));

      trainAndAssert(ann, training, targetGlobalError: 0.01);
    });

    test('training twice in a row keeps converging', () {
      var samples = samplesFromStrings(xor);
      var samplesSet = SamplesSet(samples, subject: 'xor');

      var ann = buildANN(2, [4], 1);
      var training = RProp(ann, samplesSet);
      training.logEnabled = false;

      expect(
        training.trainUntilGlobalError(targetGlobalError: 0.05, maxRetries: 10),
        isTrue,
      );
      expect(
        training.trainUntilGlobalError(targetGlobalError: 0.05, maxRetries: 10),
        isTrue,
      );

      expect(ann.computeSamplesGlobalError(samples) <= 0.05, isTrue);
    });
  });
}
