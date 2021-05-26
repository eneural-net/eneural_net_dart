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

  var samples = SampleFloat32x4.toListFromString([
    '0,0=0',
    '1,0=1',
    '0,1=1',
    '1,1=0',
  ], scale, true);

  var samplesSet = SamplesSet(samples, subject: 'xor');

  var ann = ANN(
      scale,
      LayerFloat32x4(2, true, ActivationFunctionLinear()),
      [HiddenLayerConfig(3, true, activationFunction)],
      LayerFloat32x4(1, false, activationFunction));

  print(ann);

  var backpropagation = Backpropagation(ann, samplesSet);

  var chronometer = Chronometer('Backpropagation').start();

  while (chronometer.operations < limit) {
    backpropagation.train(100, 0.01);
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
