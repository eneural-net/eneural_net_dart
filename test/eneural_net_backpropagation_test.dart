import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/src/eneural_net_training.dart';
import 'package:test/test.dart';

void main() {
  group('Basic ANNs', () {
    setUp(() {
      print('================================================================');
    });

    test('Activate Test - SampleFloat32', () {
      var scale = ScaleDouble.ZERO_TO_ONE;

      var samples = SampleFloat32.toListFromString([
        '0,0=0',
        '0,1=1',
        '1,0=1',
        '1,1=0',
      ], scale, true);

      var activationFunction = ActivationFunctionSigmoidBoundedFast();
      var ann = ANN(scale, LayerFloat32(2, activationFunction), [3],
          LayerFloat32(1, activationFunction));

      print(ann);

      var chronometerActivation = Chronometer('Activation').start();

      chronometerActivation.start();

      for (var i = 0; i < 100000; ++i) {
        for (var sample in samples) {
          ann.activate(sample.input);
          chronometerActivation.operations++;
        }
      }

      chronometerActivation.stop();

      print(chronometerActivation);

      print('Train...');

      var training = Backpropagation(ann, learningRate: 0.25);

      var chronometer = Chronometer('Backpropagation').start();

      for (var i = 0; i < 200; ++i) {
        var globalError = training.train(samples, 100);
        chronometer.operations += 10;

        if (i % 100 == 0) {
          print('$i> globalError: $globalError');
        }
      }

      chronometer.stop();

      print(ann);

      for (var sample in samples) {
        ann.activate(sample.input);
        print(
            '- ${sample.input.values} -> ${ann.output} (${sample.output.values})');
      }

      print(chronometer);
    });
  });
}
