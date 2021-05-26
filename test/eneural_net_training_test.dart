import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:test/test.dart';

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;
  var samplesXOR_Float32x4 = SampleFloat32x4.toListFromString([
    '0,0=0',
    '0,1=1',
    '1,0=1',
    '1,1=0',
  ], scaleDouble, true);

  group('Training', () {
    setUp(() {
      print('================================================================');
    });

    test('Backpropagation + ActivationFunctionSigmoid', () {
      var activationFunction = ActivationFunctionSigmoid();

      var ann = ANN(
          scaleDouble,
          LayerFloat32x4(2, true, activationFunction),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false, activationFunction));

      var training = Backpropagation(
          ann, SamplesSet(samplesXOR_Float32x4, subject: 'xor'));

      _trainANN(ann, training);
    });

    test('RProp + ActivationFunctionSigmoid', () {
      var activationFunction = ActivationFunctionSigmoid();

      var ann = ANN(
          scaleDouble,
          LayerFloat32x4(2, true, activationFunction),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false, activationFunction));

      var training =
          RProp(ann, SamplesSet(samplesXOR_Float32x4, subject: 'xor'));

      _trainANN(ann, training);
    });

    test('Backpropagation + ActivationFunctionSigmoidFast', () {
      var activationFunction = ActivationFunctionSigmoidFast();

      var ann = ANN(
          scaleDouble,
          LayerFloat32x4(2, true, activationFunction),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false, activationFunction));

      var training = Backpropagation(
          ann, SamplesSet(samplesXOR_Float32x4, subject: 'xor'));

      _trainANN(ann, training);
    });

    test('Backpropagation + ActivationFunctionSigmoidBoundedFast', () {
      var activationFunction = ActivationFunctionSigmoidBoundedFast();

      var ann = ANN(
          scaleDouble,
          LayerFloat32x4(2, true, activationFunction),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false, activationFunction));

      var training = Backpropagation(
          ann, SamplesSet(samplesXOR_Float32x4, subject: 'xor'));

      _trainANN(ann, training);
    });
  });
}

void _trainANN<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>,
        P extends Sample<N, E, T, S>>(
    ANN<N, E, T, S> ann, Training<N, E, T, S, P> training) {
  print(ann);

  print('Train...');

  var chronometer = Chronometer(training.algorithmName).start();

  var ok =
      training.trainUntilGlobalError(targetGlobalError: 0.05, maxRetries: 10);

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
        '- ${sample.input.values} -> ${ann.output} (${sample.output.values}) ; error: $sampleError $sampleError');

    expect(sampleError < 0.20, isTrue);
  }

  print(chronometer);
}
