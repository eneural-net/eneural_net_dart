import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

import 'package:eneural_net/src/eneural_net_extension.dart';

void main() {
  runAnnFloat32x4(4000000, ActivationFunctionSigmoid());
  runAnnFloat32x4(4000000, ActivationFunctionSigmoid());
  runAnnFloat32x4(4000000, ActivationFunctionSigmoid());

  print('------------------\n');

  runAnnFloat32x4(6000000, ActivationFunctionSigmoidFast());
  runAnnFloat32x4(6000000, ActivationFunctionSigmoidFast());
  runAnnFloat32x4(6000000, ActivationFunctionSigmoidFast());

  print('------------------\n');

  runAnnFloat32x4(6000000, ActivationFunctionSigmoidBoundedFast());
  runAnnFloat32x4(6000000, ActivationFunctionSigmoidBoundedFast());
  runAnnFloat32x4(6000000, ActivationFunctionSigmoidBoundedFast());
}

Chronometer runAnnFloat32x4(
    int limit, ActivationFunction<double, Float32x4> activationFunction) {
  var scale = ScaleDouble.ZERO_TO_ONE;

  var samples = SampleFloat32.toListFromString([
    '0,0=0',
    '1,0=1',
    '0,1=1',
    '1,1=0',
  ], scale, true);

  var ann = ANN(scale, LayerFloat32(2, activationFunction), [3],
      LayerFloat32(1, activationFunction));

  print(ann);

  var backpropagation = Backpropagation(ann);

  var chronometer = Chronometer('Backpropagation').start();

  while (chronometer.operations < limit) {
    backpropagation.train(samples, 100);
    chronometer.operations += samples.length * 100;
  }

  chronometer.stop();

  var globalError = ann.computeSamplesGlobalError(samples);

  for (var i = 0; i < samples.length; ++i) {
    var sample = samples[i];

    var input = sample.input;
    var expected = sample.output;
    ann.activate(input);
    var output = ann.output;

    print('$i> $input -> $output ($expected) > error: ${output - expected}');
  }

  print('globalError: $globalError');

  print(chronometer);
  print('');

  return chronometer;
}
