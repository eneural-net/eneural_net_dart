# eneural_net

[![pub package](https://img.shields.io/pub/v/eneural_net.svg?logo=dart&logoColor=00b9fc)](https://pub.dev/packages/eneural_net)
[![Null Safety](https://img.shields.io/badge/null-safety-brightgreen)](https://dart.dev/null-safety)

[![CI](https://img.shields.io/github/workflow/status/eneural-net/eneural_net_dart/Dart%20CI/master?logo=github-actions&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/actions)
[![GitHub Tag](https://img.shields.io/github/v/tag/eneural-net/eneural_net_dart?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/releases)
[![New Commits](https://img.shields.io/github/commits-since/eneural-net/eneural_net_dart/latest?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/network)
[![Last Commits](https://img.shields.io/github/last-commit/eneural-net/eneural_net_dart?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/commits/master)
[![Pull Requests](https://img.shields.io/github/issues-pr/eneural-net/eneural_net_dart?logo=github&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/pulls)
[![Code size](https://img.shields.io/github/languages/code-size/eneural-net/eneural_net_dart?logo=github&logoColor=white)](https://github.com/eneural-net/eneural_net_dart)
[![License](https://img.shields.io/github/license/eneural-net/eneural_net_dart?logo=open-source-initiative&logoColor=green)](https://github.com/eneural-net/eneural_net_dart/blob/master/LICENSE)

[eNeural.net / Dart][eNeural.net] is an AI Library for efficient Artificial Neural Networks.
The library is portable (native, JS/Web, Flutter) and the computation
is capable to use SIMD (Single Instruction Multiple Data) to improve performance.

## Usage

```dart
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
```

Output:

```text
ANN<double, Float32x4, SignalFloat32x4, Scale<double>>{ layers: 2+ -> [3+] -> 1 ; ScaleDouble{0.0 .. 1.0}  ; ActivationFunctionSigmoid }
Backpropagation<double, Float32x4, SignalFloat32x4, Scale<double>, SampleFloat32x4>{name: Backpropagation}

---------------------------------------------------
Backpropagation> [INFO] Started Backpropagation training session "xor". { samples: 4 ; targetGlobalError: 0.01 }
Backpropagation> [INFO] Selected initial ANN from poll of size 100, executing 600 epochs. Lowest error: 0.2451509315860858 (0.2479563313068569)
Backpropagation> [INFO] (OK) Reached target error in 2317 epochs (107 ms). Final error: 0.009992250436771877 <= 0.01
---------------------------------------------------

Samples Outputs:
- 0> [0, 0] -> [0.11514352262020111] ([0]) > error: [0.11514352262020111]
- 1> [1, 0] -> [0.9083549976348877] ([1]) > error: [-0.0916450023651123]
- 2> [0, 1] -> [0.9032943248748779] ([1]) > error: [-0.09670567512512207]
- 3> [1, 1] -> [0.09465821087360382] ([0]) > error: [0.09465821087360382]

globalError: 0.009992250436771877
achievedTargetError: true

Backpropagation{elapsedTime: 111 ms, hertz: 83495.49549549549 Hz, ops: 9268, startTime: 2021-05-26 06:25:34.825383, stopTime: 2021-05-26 06:25:34.936802}
```

# SIMD (Single Instruction Multiple Data)

Dart has support for SIMD when computation is made using [Float32x4] and [Int32x4].

The basic principle with SIMD is to execute math operations simultaneously in 4 numbers.

[Float32x4] is a lane of 4 [double] (32 bits single precision floating points).
Example of multiplication:

```dart
  var fs1 = Float32x4( 1.1 , 2.2 , 3.3  , 4.4  );
  var fs2 = Float32x4( 10  , 100 , 1000 , 1000 );
  
  var fs3 = fs1 * fs2 ;
  
  print(fs3);
  // Output:
  // [11.000000, 220.000000, 3300.000000, 4400.000000]
```

See "[dart:typed_data library][dart_typed_data]" and "[Using SIMD in Dart][using_simd]".

[double]: https://api.dart.dev/stable/2.12.4/dart-core/double-class.html
[Float32x4]: https://api.dart.dev/stable/2.12.4/dart-typed_data/Float32x4-class.html
[Int32x4]: https://api.dart.dev/stable/2.12.4/dart-typed_data/Int32x4-class.html
[dart_typed_data]: https://api.dart.dev/stable/2.12.4/dart-typed_data/dart-typed_data-library.html
[using_simd]: https://www.dartcn.com/articles/server/simd

# Signal

The class `Signal` represents the collection of numbers (including its related operations)
that will flow through the `ANN`, representing the actual signal that
an Artificial Neural Network should compute.

The main implementation is `SignalFloat32x4` and represents
an `ANN` `Signal` based in [Float32x4]. All the operations prioritizes the use of SIMD.

The framework of `Signal` allows the implementation of any kind of data
to represent the numbers and operations of an [eNeural.net] `ANN`. `SignalInt32x4`
is an experimental implementation to exercise an `ANN` based in integers.

# Activation Functions

`ActivationFunction` is the base class for `ANN` neurons activation functions:

- `ActivationFunctionSigmoid`:
  
  The classic Sigmoid function (return for `x` a value between `0.0` and `1.0`):
  
  ```dart
  activation(double x) {
    return 1 / (1 + exp(-x)) ;
  }
  ```
  


- `ActivationFunctionSigmoidFast`:
  
  Fast approximation version of Sigmoid function, that is not based in [exp(x)][dart_math_exp]:

  ```dart
  activation(double x) {
    x *= 3 ;
    return 0.5 + ((x) / (2.5 + x.abs()) / 2) ;
  }
  ```
  Function author: Graciliano M. Passos: [gmpassos@GitHub][github].  


- `ActivationFunctionSigmoidBoundedFast`:

  Fast approximation version of Sigmoid function, that is not based in [exp(x)][dart_math_exp],
  bounded to a lower and upper limit for [x].

  ```dart
  activation(double x) {
    if (x < lowerLimit) {
      return 0.0 ;
    } else if (x > upperLimit) {
      return 1.0 ;
    }
    x = x / scale ;
    return 0.5 + (x / (1 + (x * x))) ;
  }
  ```

  Function author: Graciliano M. Passos: [gmpassos@GitHub][github].

## exp(x)

[exp][dart_math_exp] is the function of the natural exponent,
[e][dart_math_e], to the power x.

This is an important `ANN` function, since is used by the popular
Sigmoid function, and usually a high precision version is slow,
but a high precision version is actually not necessary
for Artificial Neural Networks, opening the opportunity
for implementations that are just an approximation.

[dart_math_e]: https://api.dart.dev/stable/2.12.1/dart-math/e-constant.html
[dart_math_exp]: https://api.dart.dev/stable/2.12.1/dart-math/exp.html

# Fast Math

An internal fast math library is used to compute `ActivationFunctionSigmoid`.

If you want you can import this library and use it in your projects:

```dart
import 'package:eneural_net/eneural_net_fast_math.dart' as fast_math ;

void main() {
  // Fast `exp` function:
  fast_math.exp(2);

  // Fast high precision `exp` function:
  var highPrecision = <double>[0.0 , 0.0];
  fast_math.expHighPrecision(2, 0.0, highPrecision);
}
```

The implementation is based in the Dart package [Complex](https://pub.dev/packages/complex):
- https://github.com/rwl/complex/blob/master/lib/src/fastmath.dart

# eNeural.net

You can find more at: [eneural.net][eNeural.net]

[eNeural.net]: https://eneural.net/

## Source

The official source code is [hosted @ GitHub][github_eneural_net]:

- https://github.com/eneural-net/eneural_net_dart

[github_eneural_net]: https://github.com/eneural-net/eneural_net_dart

# Features and bugs

Please file feature requests and bugs at the [issue tracker][tracker].

# Contribution

Any help from the open-source community is always welcome and needed:
- Found an issue?
    - Please fill a bug report with details.
- Wish a feature?
    - Open a feature request with use cases.
- Are you using and liking the project?
    - Promote the project: create an article, do a post or make a donation.
- Are you a developer?
    - Fix a bug and send a pull request.
    - Implement a new feature, like other training algorithms and activation functions.
    - Improve the Unit Tests.
- Have you already helped in any way?
    - **Many thanks from me, the contributors and everybody that uses this project!**


[tracker]: https://github.com/eneural-net/eneural_net_dart/issues

# Author

Graciliano M. Passos: [gmpassos@GitHub][github].

[github]: https://github.com/gmpassos

## License

[Apache License - Version 2.0][apache_license]

[apache_license]: https://www.apache.org/licenses/LICENSE-2.0.txt
