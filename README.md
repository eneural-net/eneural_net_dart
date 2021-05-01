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

[eNeural.net] is an AI Library for efficient Artificial Neural Networks.
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

  // The activation function to use in the ANN:
  var activationFunction = ActivationFunctionSigmoid();

  // The ANN using layers that can compute with Float32x4 (SIMD compatible type).
  var ann = ANN(
    scale,
    LayerFloat32x4(2, activationFunction), // Input layer: 2 neurons
    [3],                                   // 1 Hidden layer: 3 neurons
    LayerFloat32x4(1, activationFunction), // Output layer: 1 neuron
  );

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

# eNeural.net

You can find more at: [eneural.net][eNeural.net]

[eNeural.net]: https://eneural.net/

# Features and bugs

Please file feature requests and bugs at the [issue tracker][tracker].

# Contribution

Any help from open-source community is always welcome and needed:
- Found an issue?
    - Fill a bug with details.
- Wish a feature?
    - Open a feature request.
- Are you using and liking the project?
    - Promote the project: create an article, post or make a donation.
- Are you a developer?
    - Fix a bug and send a pull request.
    - Implement a new feature, like other training algorithms and activation functions.
    - Improve unit tests.
- Have you already helped in any way?
    - **Many thanks from me, the contributors and everybody that uses this project!**


[tracker]: https://github.com/eneural-net/eneural_net_dart/issues

# Author

Graciliano M. Passos: [gmpassos@GitHub][github].

[github]: https://github.com/gmpassos

## License

[Apache License - Version 2.0][apache_license]

[apache_license]: https://www.apache.org/licenses/LICENSE-2.0.txt
