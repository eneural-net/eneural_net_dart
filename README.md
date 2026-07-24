# eneural_net

[![pub package](https://img.shields.io/pub/v/eneural_net.svg?logo=dart&logoColor=00b9fc)](https://pub.dev/packages/eneural_net)
[![Null Safety](https://img.shields.io/badge/null-safety-brightgreen)](https://dart.dev/null-safety)
[![Dart CI](https://github.com/eneural-net/eneural_net_dart/actions/workflows/dart.yml/badge.svg?branch=master)](https://github.com/eneural-net/eneural_net_dart/actions/workflows/dart.yml)
[![Acceleration CI](https://github.com/eneural-net/eneural_net_dart/actions/workflows/acceleration.yml/badge.svg?branch=master)](https://github.com/eneural-net/eneural_net_dart/actions/workflows/acceleration.yml)
[![WebGPU CI](https://github.com/eneural-net/eneural_net_dart/actions/workflows/webgpu.yml/badge.svg?branch=master)](https://github.com/eneural-net/eneural_net_dart/actions/workflows/webgpu.yml)
[![GitHub Tag](https://img.shields.io/github/v/tag/eneural-net/eneural_net_dart?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/releases)
[![New Commits](https://img.shields.io/github/commits-since/eneural-net/eneural_net_dart/latest?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/network)
[![Last Commits](https://img.shields.io/github/last-commit/eneural-net/eneural_net_dart?logo=git&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/commits/master)
[![Pull Requests](https://img.shields.io/github/issues-pr/eneural-net/eneural_net_dart?logo=github&logoColor=white)](https://github.com/eneural-net/eneural_net_dart/pulls)
[![Code size](https://img.shields.io/github/languages/code-size/eneural-net/eneural_net_dart?logo=github&logoColor=white)](https://github.com/eneural-net/eneural_net_dart)
[![License](https://img.shields.io/github/license/eneural-net/eneural_net_dart?logo=open-source-initiative&logoColor=green)](https://github.com/eneural-net/eneural_net_dart/blob/master/LICENSE)

[eNeural.net / Dart][eNeural.net] is an AI Library for efficient Artificial Neural Networks.
The library is portable (native, JS/Web, Wasm, Flutter) and the computation
is capable to use SIMD (Single Instruction Multiple Data) to improve performance.

Training can also be **GPU-accelerated**, while keeping the same pure-Dart API:
[Metal](#native-acceleration-macos-cpu--metal--windowslinux-cuda) (macOS) and
[CUDA](#native-acceleration-macos-cpu--metal--windowslinux-cuda) (Windows/Linux)
natively, and [WebGPU](#webgpu-browser-gpu) in the browser.

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
The Activation Functions are implemented using [Float32x4], improving
performance by 1.5x to 2x, when compared to normal implementation.

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

# Native Acceleration (macOS: CPU + Metal · Windows/Linux: CUDA)

The training loop can be offloaded to a native backend: on macOS via Apple
**Accelerate** (BLAS/vDSP, CPU) or **Metal** (GPU); on Windows/Linux via
**CUDA** (NVIDIA GPU). The network, its weights, the optimizer state and the full
training sample set are uploaded once and each epoch runs entirely in native code
(forward + backprop + weight update), reproducing the pure-Dart
iRProp+/Backpropagation numerics within `float32` tolerance.

Native acceleration is **optional and opt-in**. The package stays pure-Dart (it
still publishes to pub.dev and runs on the web); if the native library isn't
present, the trainers transparently fall back to the pure-Dart SIMD path.

### Build the native libraries

```bash
# macOS (CPU + Metal):
bash native/macos/build.sh
# produces (git-ignored):
#   native/macos/cpu/build/libeneural_cpu_<arch>.dylib
#   native/macos/metal/build/libeneural_metal_<arch>.dylib

# Windows/Linux (CUDA — requires the NVIDIA CUDA Toolkit: nvcc + cuBLAS):
native\cuda\build.bat        # Windows
bash native/cuda/build.sh    # Linux
# produces (git-ignored):
#   native/cuda/build/libeneural_cuda_<arch>.{dll,so}
```

### Use an accelerated trainer

`NativeRProp` and `NativeBackpropagation` are drop-in replacements for `RProp`
and `Backpropagation` on `Float32x4` networks:

```dart
import 'package:eneural_net/eneural_net.dart';

var trainer = NativeRProp(
  ann,
  SamplesSet(samples, subject: 'xor'),
  backend: NativeBackend.auto, // auto | cpu | metal | cuda | none
);

print(trainer.activeBackend); // NativeBackend.cpu (or .metal / .cuda / .none)

trainer.trainUntilGlobalError(targetGlobalError: 1e-6);
```

Backend selection:

- **`auto`** (default) — picks the best backend available for the host: on macOS
  the CPU backend for small networks and Metal (GPU) for large ones (Metal has a
  fixed per-epoch overhead, so it only overtakes the CPU backend above ~16K
  weights); on Windows/Linux the CUDA (GPU) backend when available; otherwise
  pure Dart.
- **`cpu`** *(macOS)* — Apple Accelerate (BLAS/vDSP). ~2.4x faster than pure Dart
  on the `64 -> 256 -> 16` benchmark.
- **`metal`** *(macOS)* — Apple GPU. The trainer is **batched**: all samples are
  processed at once, so each epoch is a handful of `MetalPerformanceShaders` GEMMs
  plus elementwise kernels (dispatch count is independent of the sample count). On
  the `64 -> 256 -> 16` benchmark it trains ~3.6x faster than pure Dart, and the
  lead grows with network size (~2x over the CPU backend at ~80K weights).
- **`cuda`** *(Windows/Linux)* — NVIDIA GPU. Same batched whole-epoch design as
  Metal, with the three per-layer GEMMs on **cuBLAS** and the elementwise
  activation/delta/update steps as CUDA kernels (see `native/cuda`). Requires the
  CUDA Toolkit at build time and an NVIDIA GPU at run time.
- **`none`** — forces the pure-Dart path.

Requesting a backend not available on the host (e.g. `cuda` on macOS, or `metal`
on Windows) resolves to the pure-Dart path.

Unsupported networks (integer `Int32x4` signals, or an activation function other
than `Linear`/`Sigmoid`/`SigmoidFast`/`SigmoidBoundedFast`) also fall back to
pure Dart automatically.

## WebGPU (browser GPU)

In the browser, training can run on the GPU via **WebGPU** (WGSL compute
shaders). Like the Metal backend it is a batched whole-epoch trainer, but WebGPU
is **asynchronous** (GPU readback is a `Future`), so it is exposed through
separate trainers with `Future`-returning methods rather than the synchronous
`train()`:

```dart
var trainer = WebGpuRProp(ann, samplesSet); // or WebGpuBackpropagation

if (await trainer.isWebGpuAccelerated) {
  print('training on the GPU');
}

await trainer.trainUntilGlobalErrorAsync(targetGlobalError: 1e-4);
// also: await trainer.trainAsync(epochs, targetError);
```

`WebGpuRProp` / `WebGpuBackpropagation` extend `RProp` / `Backpropagation`, so
they keep the synchronous pure-Dart API too. When WebGPU is unavailable (not a
browser, no WebGPU support, or an unsupported network) the async methods
transparently fall back to the synchronous pure-Dart trainer — so the same code
runs on the Dart VM and the web.

What runs on the device: the whole epoch. The weights, the optimizer state
(iRProp+ steps/gradients) and the full sample set stay resident in GPU buffers;
each epoch is a chain of WGSL compute passes (forward → output delta → backprop →
gradients → weight update) over every sample at once, with a single readback for
the epoch error. `activateWebGpu(input)` runs a single-sample forward pass on the
GPU (the pure-Dart `ann.activate()` keeps working as usual).

Requirements: a browser with WebGPU (Chrome/Edge 113+, Safari 26+, Firefox 141+
on supported platforms), an `ANN` on `Float32x4` signals, and activation
functions among `Linear`/`Sigmoid`/`SigmoidFast`/`SigmoidBoundedFast` — anything
else falls back to pure Dart.

See `example/eneural_net_webgpu_example.dart`.

### WebGPU tests

`test/eneural_net_webgpu_test.dart` runs everywhere (it covers the async API and
the pure-Dart fallback), while
`test/eneural_net_webgpu_integration_test.dart` is browser-only and exercises the
real GPU path: forward-pass parity with the pure-Dart `activate()` over several
topologies/activations, exact weight upload/download round-trips, and
iRProp+/Backpropagation epochs compared against the pure-Dart trainers.

The default `chrome` platform launches Chrome with `--disable-gpu`, which removes
`navigator.gpu`, so `dart_test.yaml` defines two WebGPU-enabled platforms:

```bash
# Real GPU (desktop with a GPU):
dart test test/eneural_net_webgpu_integration_test.dart -p chrome_webgpu

# Software adapter (machines/CI runners without a GPU):
dart test test/eneural_net_webgpu_integration_test.dart -p chrome_webgpu_swiftshader

# Require a WebGPU device: no device -> failure instead of skip (used by CI):
dart test test/eneural_net_webgpu_integration_test.dart -p chrome_webgpu \
  --dart2js-args=-DWEBGPU_REQUIRED=true
```

Without a WebGPU device the integration tests self-skip, so they are safe to run
anywhere. The `WebGPU CI` workflow runs them on Linux/Windows (SwiftShader) and
macOS (real GPU), always with `-DWEBGPU_REQUIRED=true` so a job cannot pass
through the fallback path, plus a job that checks the Dart VM fallback.

# Training Algorithms

Beyond `Backpropagation` and `RProp` (iRProp+), the library ships a broad set of
pure-Dart trainers (SIMD `Float32x4`):

- **Gradient optimizers** (`GradientOptimizer` seam, per-weight state, full SIMD):
  `SGD` (+ classic/Nesterov momentum), `Adam` (+ `AdamW`, `Nadam`, `AMSGrad`
  flags), `RMSProp`, `AdaGrad`, `AdaDelta`, `Quickprop`, `Lion`, and
  `ResilientPropagation` (RProp+/RProp−/iRProp+/iRProp−). All support **mini-batch
  / online** training (`batchSize`), **L2 weight decay**, **gradient clipping**,
  and pluggable **learning-rate schedules** (`StepDecayStrategy`,
  `ExponentialDecayStrategy`, `CosineAnnealingStrategy`, `WarmupStrategy`).
- **Second-order** (finite-difference based, ideal for small nets):
  `ConjugateGradient`, `LBFGS`, `LevenbergMarquardt`.
- **Population / gradient-free**: `EvolutionStrategy`, `SeparableCMAES`,
  `GeneticAlgorithm`, `ParticleSwarm`, `DifferentialEvolution`,
  `SimulatedAnnealing`.
- **Regularization**: dropout (per hidden layer via `HiddenLayerConfig`), L2 /
  weight decay, gradient clipping.

```dart
var adam = Adam(ann, samplesSet, learningRate: 0.001, weightDecay: 0.01);
adam.trainUntilGlobalError(targetGlobalError: 1e-4);
```

### Selecting by name and checkpointing

Trainers can be built by name and resumed from a JSON checkpoint:

```dart
var trainer = trainingByName('adam', ann, samplesSet, params: {'learningRate': 0.01});

// Save / restore (JSON-serializable) — ANN weights + optimizer state:
var checkpoint = saveTrainingCheckpoint(trainer);
restoreTrainingCheckpoint(trainer, checkpoint); // into a same-config trainer

print(registeredTrainings()); // adam, adamw, nadam, rmsprop, lion, lbfgs, ...
```

See `example/eneural_net_optimizers_example.dart`.

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
Sigmoid function, and usually a high precision version is slow
and approximation versions can be used for most `ANN` models and training
algorithms.

[dart_math_e]: https://api.dart.dev/stable/2.12.1/dart-math/e-constant.html
[dart_math_exp]: https://api.dart.dev/stable/2.12.1/dart-math/exp.html

# Fast Math

An internal Fast Math library is present and can be used for platforms
that are not efficient to compute `exp` (Exponential function).

You can import this library and use it to create a specialized
`ActivationFunction` implementation or use it in any kind of project:

```dart
import 'package:eneural_net/eneural_net_fast_math.dart' as fast_math ;

void main() {
  // Fast Exponential function:
  var o = fast_math.exp(2);

  // Fast Exponential function with high precision:
  var highPrecision = <double>[0.0 , 0.0];
  var oHighPrecision = fast_math.expHighPrecision(2, 0.0, highPrecision);
  
  // Fast Exponential function with SIMD acceleration:
  var o32x4 = fast_math.expFloat32x4( Float32x4(2,3,4,5) );
}
```

The implementation is based in the Dart package [Complex](https://pub.dev/packages/complex):
- https://github.com/rwl/complex/blob/master/lib/src/fastmath.dart

The `fast_math.expFloat32x4` function was created by Graciliano M. Passos ([gmpassos@GitHub][github]).

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
