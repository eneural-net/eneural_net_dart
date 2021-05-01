import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/src/eneural_net_training.dart';
import 'package:test/test.dart';

void main() {
  group('Training', () {
    setUp(() {
      print('================================================================');
    });

    test('Backpropagation', () {
      var scale = ScaleDouble.ZERO_TO_ONE;

      var samples = SampleFloat32x4.toListFromString([
        '0,0=0',
        '0,1=1',
        '1,0=1',
        '1,1=0',
      ], scale, true);

      var activationFunction = ActivationFunctionSigmoidBoundedFast();

      var ann = ANN(scale, LayerFloat32x4(2, activationFunction), [3],
          LayerFloat32x4(1, activationFunction));

      print(ann);

      print('Train...');

      var training = Backpropagation(ann, learningRate: 0.25);

      var chronometer = Chronometer('Backpropagation').start();

      var ok = training.trainUntilGlobalError(samples,
          targetGlobalError: 0.10, maxRetries: 10);

      chronometer.stop(operations: training.totalTrainingActivations);

      print(ann);

      expect(ok, isTrue);

      expect(ann.computeSamplesGlobalError(samples) < 0.10, isTrue);

      for (var sample in samples) {
        ann.activate(sample.input);
        print(
            '- ${sample.input.values} -> ${ann.output} (${sample.output.values})');
      }

      print(chronometer);
    });
  });
}
