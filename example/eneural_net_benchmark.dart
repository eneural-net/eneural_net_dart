import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/src/eneural_net_extension.dart';

void main() {
  var totalOperations = 4000000;

  var allBenchmarks = <Chronometer?>[];

  for (var i = 0; i < 2; ++i) {
    print('\n==========================================================\n');

    var benchmarks = <Chronometer>[];

    benchmark(
      benchmarks,
      3,
      () => runAnnFloat32x4(totalOperations, ActivationFunctionSigmoid()),
    );

    print('------------------\n');

    benchmark(
      benchmarks,
      3,
      () => runAnnFloat32x4(totalOperations, ActivationFunctionSigmoidFast()),
    );

    print('------------------\n');

    benchmark(
      benchmarks,
      3,
      () => runAnnFloat32x4(
        totalOperations,
        ActivationFunctionSigmoidBoundedFast(),
      ),
    );

    print('----------------------------------------------------------\n');

    for (var session in benchmarks) {
      print(session);
    }

    allBenchmarks.addAll(benchmarks);
    allBenchmarks.add(null);
  }

  print('==========================================================');

  for (var session in allBenchmarks) {
    print(session ?? '--------------------');
  }
}

void benchmark(
  List<Chronometer> allBenchmarks,
  int sessions,
  Chronometer Function() runner,
) {
  var results = <Chronometer>[];

  for (var i = 0; i < sessions; ++i) {
    var chronometer = runner();
    results.add(chronometer);
  }

  results.sort();
  allBenchmarks.add(results.last);
}

Chronometer runAnnFloat32x4(
  int limit,
  ActivationFunction<double, Float32x4> activationFunction,
) {
  var scale = ScaleDouble.ZERO_TO_ONE;

  var samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '1,0=1', '0,1=1', '1,1=0'],
    scale,
    true,
  );

  var samplesSet = SamplesSet(
    samples,
    subject: 'xor[${activationFunction.name}]',
  );

  var ann = ANN(
    scale,
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(3, true, activationFunction)],
    LayerFloat32x4(1, false, activationFunction),
  );

  print(ann);

  var training = Backpropagation(ann, samplesSet);
  training.logProgressEnabled = true;

  var title =
      '${training.algorithmName}/${samplesSet.subject}/$activationFunction';

  var chronometer = Chronometer(title).start();

  while (chronometer.operations < limit) {
    training.train(100, 0.000001);
    chronometer.operations = training.totalTrainingActivations;
    //print(chronometer.operations);
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
