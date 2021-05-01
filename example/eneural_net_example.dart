import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

void main() {
  // Type of scale to use to compute the ANN:
  var scale = ScaleDouble.ZERO_TO_ONE;

  // The samples to learn
  var samples = SampleFloat32x4.toListFromString(
    [
      '0,0=0',
      '1,0=1',
      '0,1=1',
      '1,1=0',
    ],
    scale,
    true, // Already normalized in the scale.
  );

  // The activation function to use in the ANN:
  var activationFunction = ActivationFunctionSigmoid();

  // The ANN using layers that can compute with Float32x4 (SIMD compatible type).
  var ann = ANN(scale, LayerFloat32x4(2, activationFunction), [3],
      LayerFloat32x4(1, activationFunction));

  print(ann);

  // Training algorithm:
  var backpropagation = Backpropagation(ann);

  var chronometer = Chronometer('Backpropagation').start();

  // Train the ANN using Backpropagation until global error 0.01,
  // with max epochs per training session of 1000000 and
  // a max retry of 10 when a training session can't reach
  // the target global error:
  var achievedTargetError = backpropagation.trainUntilGlobalError(samples,
      targetGlobalError: 0.01, maxEpochs: 1000000, maxRetries: 10);

  chronometer.stop(operations: backpropagation.totalTrainingActivations);

  // Compute the current global error of the ANN:
  var globalError = ann.computeSamplesGlobalError(samples);

  for (var i = 0; i < samples.length; ++i) {
    var sample = samples[i];

    var input = sample.input;
    var expected = sample.output;

    // Activate the sample input:
    ann.activate(input);

    // The current output of the ANN (after activation):
    var output = ann.output;

    print('- $i> $input -> $output ($expected) > error: ${output - expected}');
  }

  print('globalError: $globalError');
  print('achievedTargetError: $achievedTargetError');

  print(chronometer);
}
