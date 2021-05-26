import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

void main() {
  // Type of scale to use to compute the ANN:
  var scale = ScaleDouble.ZERO_TO_ONE;

  // The samples to learn in Float32x4 data type:
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

  var samplesSet = SamplesSet(samples, subject: 'xor');

  // The activation function to use in the ANN:
  var activationFunction = ActivationFunctionSigmoid();

  // The ANN using layers that can compute with Float32x4 (SIMD compatible type).
  var ann = ANN(
    scale,
    // Input layer: 2 neurons with linear activation function:
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    // 1 Hidden layer: 3 neurons with sigmoid activation function:
    [HiddenLayerConfig(3, true, activationFunction)],
    // Output layer: 1 neuron with sigmoid activation function:
    LayerFloat32x4(1, false, activationFunction),
  );

  print(ann);

  // Training algorithm:
  var backpropagation = Backpropagation(ann, samplesSet);

  print(backpropagation);

  print('\n---------------------------------------------------');

  var chronometer = Chronometer('Backpropagation').start();

  // Train the ANN using Backpropagation until global error 0.01,
  // with max epochs per training session of 1000000 and
  // a max retry of 10 when a training session can't reach
  // the target global error:
  var achievedTargetError = backpropagation.trainUntilGlobalError(
      targetGlobalError: 0.01, maxEpochs: 50000, maxRetries: 10);

  chronometer.stop(operations: backpropagation.totalTrainingActivations);

  print('---------------------------------------------------\n');

  // Compute the current global error of the ANN:
  var globalError = ann.computeSamplesGlobalError(samples);

  print('Samples Outputs:');
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

  print('\nglobalError: $globalError');
  print('achievedTargetError: $achievedTargetError\n');

  print(chronometer);
}
