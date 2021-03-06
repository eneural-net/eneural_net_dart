import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

void main() async {
  var totalOperations = 40000000;

  var activationFunctions = <ActivationFunction<double, Float32x4>>[
    ActivationFunctionSigmoid(),
    ActivationFunctionSigmoidFast(),
    ActivationFunctionSigmoidBoundedFast(),
  ];

  var allBenchmarks = <Chronometer?>[];

  for (var i = 0; i < 3; ++i) {
    print('\n==========================================================\n');

    var benchmarks = <Chronometer>[];

    for (var af in activationFunctions) {
      var bestResult =
          benchmark(benchmarks, 3, () => runAnnFloat32x4(totalOperations, af));

      print('bestResult: $bestResult > $af');
      print('------------------\n');
    }

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

Chronometer benchmark(List<Chronometer> allBenchmarks, int sessions,
    Chronometer Function() runner) {
  var results = <Chronometer>[];

  for (var i = 0; i < sessions; ++i) {
    var chronometer = runner();
    results.add(chronometer);
  }

  results.sort();

  var bestResult = results.last;
  allBenchmarks.add(bestResult);

  return bestResult;
}

var in1 = Float32x4(0.0, 0.25, 0.50, 1.0);
var in2 = Float32x4(1.0, 0.50, 0.25, 0.0);
var in3 = Float32x4(-6.0, -3.0, 3.0, 6.0);
var in4 = Float32x4(-2.0, -1.0, 1.0, 2.0);
var in5 = Float32x4(-12.0, -6.0, 6.0, 12.0);

Chronometer runAnnFloat32x4(
    int limit, ActivationFunction<double, Float32x4> activationFunction) {
  print(activationFunction);

  var chronometer = Chronometer(activationFunction.name).start();

  var result = in1;

  while (chronometer.operations < limit) {
    result = result * activationFunction.activateEntry(in1);
    result = result * activationFunction.activateEntry(in2);
    result = result * activationFunction.activateEntry(in3);
    result = result * activationFunction.activateEntry(in4);
    result = result * activationFunction.activateEntry(in5);
    chronometer.operations += 5;
  }

  chronometer.stop();

  print('result: $result');
  print(chronometer);
  print('');

  return chronometer;
}
