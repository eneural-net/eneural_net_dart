import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:test/test.dart';

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;
  var samplesXorFloat32x4 = SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scaleDouble,
    true,
  );

  /// Builds the XOR ANN. [seed] keeps the training reproducible.
  ANN<double, Float32x4, SignalFloat32x4, Scale<double>> buildANN(
    ActivationFunction<double, Float32x4> activationFunction, {
    int hidden = 3,
    int seed = 101,
  }) => ANN(
    scaleDouble,
    LayerFloat32x4(2, true, activationFunction),
    [HiddenLayerConfig(hidden, true)],
    LayerFloat32x4(1, false, activationFunction),
    random: Random(seed),
  );

  group('Training', () {
    setUp(() {
      print('================================================================');
    });

    test('Backpropagation + ActivationFunctionSigmoid', () {
      var ann = buildANN(ActivationFunctionSigmoid());

      var training = Backpropagation(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('RProp + ActivationFunctionSigmoid', () {
      var ann = buildANN(ActivationFunctionSigmoid());

      var training = RProp(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('Backpropagation + ActivationFunctionSigmoidFast', () {
      var ann = buildANN(ActivationFunctionSigmoidFast());

      var training = Backpropagation(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('RProp + ActivationFunctionSigmoidFast', () {
      var ann = buildANN(ActivationFunctionSigmoidFast(), hidden: 4);

      var training = RProp(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('Backpropagation + ActivationFunctionSigmoidBoundedFast', () {
      var ann = buildANN(ActivationFunctionSigmoidBoundedFast());

      var training = Backpropagation(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('RProp + ActivationFunctionSigmoidBoundedFast', () {
      var ann = buildANN(ActivationFunctionSigmoidBoundedFast(), hidden: 4);

      var training = RProp(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });

    test('Backpropagation + ActivationFunctionLinear input layer', () {
      // A `Linear` input layer with `Sigmoid` hidden/output layers:
      var ann = ANN(
        scaleDouble,
        LayerFloat32x4(2, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(4, true, ActivationFunctionSigmoid())],
        LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
        random: Random(101),
      );

      var training = RProp(
        ann,
        SamplesSet(samplesXorFloat32x4, subject: 'xor'),
      );

      _trainANN(ann, training);
    });
  });
}

void _trainANN<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>(ANN<N, E, T, S> ann, Training<N, E, T, S, P> training) {
  print(ann);

  print('Train...');

  var chronometer = Chronometer(training.algorithmName).start();

  var ok = training.trainUntilGlobalError(
    targetGlobalError: 0.05,
    maxRetries: 20,
    random: Random(101),
  );

  chronometer.stop(operations: training.totalTrainingActivations);

  print(ann);

  expect(ok, isTrue);

  var globalError = ann.computeSamplesGlobalError(training.samples);

  print('globalError: $globalError');
  expect(globalError <= 0.05, isTrue);

  for (var sample in training.samples) {
    ann.activate(sample.input);

    var sampleErrors = ann.output - sample.output;
    var sampleError = sampleErrors.squaresMean;

    print(
      '- ${sample.input.values} -> ${ann.output} (${sample.output.values}) ; error: $sampleError $sampleError',
    );

    expect(sampleError < 0.20, isTrue);
  }

  print(chronometer);
}
