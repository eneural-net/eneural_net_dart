## 1.3.2

- **FIX (Backpropagation/RProp)**: bias neurons propagate a constant `1` in the
  forward pass, but the gradient computation used the value stored in the bias
  neuron slot instead. That value is `0` for the input layer and `f(net)` for
  hidden layers, so:
  - **Input-layer bias weights received a zero gradient and never learned** —
    hidden neurons had no learnable input threshold (every hidden neuron output
    was pinned to `0.5` for a zero input).
  - Hidden-layer bias weights learned with a wrong-magnitude (but correct-sign)
    gradient.

  The gradient now uses the bias neuron's true forward output (`1`). Verified
  by numerical gradient checking: the analytical gradient now matches the
  central-difference gradient for **every** weight (cosine `0.933 → 1.000`),
  and networks converge faster and to a lower error.

  Added `test/eneural_net_backprop_gradient_test.dart` (gradient-check harness).

## 1.3.1

- Test suite expanded from 24 to 565 tests, including integration tests.
  Line coverage: 84% -> 99.7%.

- `Signal`:
  - **FIX**: `lastEntryLength` returned a negative value for the
    implementations that allocate entries in chunks of 4 (`SignalInt32x4` and
    `SignalFloat32x4Mod4`), breaking `computeSumSquares`.
  - **FIX**: `setExtraValues` threw `StateError` whenever the padding was
    bigger than 3 values, which made **any `Int32x4` ANN impossible to build**.
  - **FIX**: `SignalInt32x4.from`/`fromEntries` and
    `SignalFloat32x4Mod4.fromEntries` produced signals whose entries length
    was not a multiple of 4, breaking their unrolled SIMD loops.
  - **FIX**: `hashCode` was identity based while `==` was value based, so
    equal signals could not be used as `Set`/`Map` keys.
  - **FIX**: `SignalFloat32x4Mod4.copy()` returned a plain `SignalFloat32x4`.
  - **FIX**: `SignalFloat32x4Mod4.calcEntriesCapacityForSize` ignored the
    chunking.
  - **FIX**: `multiply`, `subtract` and `multiplyEntries` returned a signal of
    `capacity` length instead of `length`.
  - Added `valuesEntriesLength`: the number of entries that hold values.

- `Scale`:
  - **FIX**: `ScaleZoomableInt` could not be decoded from JSON (the emitted
    format name didn't match `Scale.fromJson`, and `zoom` was never
    serialized).

- `ActivationFunction`:
  - **FIX**: `ActivationFunctionSigmoidBoundedFast.activateEntry` (SIMD)
    computed a different function than `activate`, and ignored `scale`.
  - **FIX**: `ActivationFunctionSigmoidFast` and
    `ActivationFunctionSigmoidBoundedFast` added the `flat spot` in the SIMD
    derivative but not in the scalar one.
  - **FIX**: `ActivationFunctionSigmoidFastInt.derivativeEntry` used a
    hardcoded `100` instead of `scaleMax`.
  - **FIX**: `createRandomWeights` ignored its `scale` argument.
  - **FIX**: `byName`/`fromJson` didn't know the `Int32x4` functions, making
    the `Int32x4` branches of `ANN.fromJson` unreachable. Added `scaleMax` to
    `byName` and to `ActivationFunctionSigmoidFastInt.toJsonMap`.

- `ANN`/`Layer`:
  - **FIX**: the "no bias neuron at the output layer" check never fired,
    silently adding an extra output neuron.
  - **FIX**: `Layer.toJsonMap()` threw on a layer that was not connected yet.
  - **FIX**: `resetWeights` left the padding weights with random values.

- `Sample`:
  - **FIX**: `proximityStatistics` used the input proximity twice, ignoring
    the output.
  - **FIX**: `SamplesSet.samplesSimilarityGroups` threw `RangeError` when
    given a `samples` list shorter than the set.

- `Training`:
  - **FIX**: the `subject` parameter of `Training`/`Backpropagation`/`RProp`
    was discarded.
  - **FIX**: `LearningRateStrategy` could never recover the learning rate
    (the counter was reset on every call), and computed an `Infinity` initial
    value before the training was initialized.
  - Added `bestTrainingError` and `resetBestTraining`: a new training session
    no longer inherits the best weights of the previous one.

- `DataStatistics`:
  - **FIX**: `standardDeviation` was the RMS (the mean was never subtracted),
    disagreeing with `List.standardDeviation`.
  - **FIX**: `mean`/`standardDeviation`/`squaresMean` were `NaN` for an empty
    series.
  - Added `computeStandardDeviation` and `computeSquaresSum`.

- `Chronometer`:
  - **FIX**: `reset()` didn't reset `failedOperations`.

- `fast_math`:
  - **FIX**: `expm1` never wrote its high precision output, which made
    `sinh(x)` return `0.0` for every `|x| <= 0.25`.
  - **FIX**: `copySign` ignored the sign bit of `-0.0`, so `atan2(-0.0, x)`
    with a negative `x` returned `+pi` instead of `-pi`.
  - **FIX**: `exp`, `expHighPrecision`, `expm1` and `atan` threw
    `UnsupportedError` or returned a wrong finite value for `NaN`/infinities.

- Extensions:
  - `allEquals` on an empty collection is now vacuously `true`.
  - Added `List.plus` (element-wise sum): the `+` operator of the extensions
    is shadowed by `List.operator +` (concatenation) and can never be reached
    through the operator syntax. The `clamp` extensions are likewise shadowed
    by `num.clamp` and are now documented as such.

## 1.3.0

- Code reformatted with the new Dart formatter style (no behavior changes).

- sdk: '>=3.10.0 <4.0.0'
- collection: ^1.19.1
- swiss_knife: ^3.1.6
- test: ^1.31.2

## 1.2.0

- Optimize & update Dart CI.

- sdk: '>=3.0.0 <4.0.0'
- collection: ^1.17.2
- swiss_knife: ^3.1.5
- intl: ^0.18.1
- lints: ^2.1.1
- test: ^1.24.6
- dependency_validator: ^3.2.3

## 1.1.3

- `ANN`:
  - Added `toJson`, `toJsonMap` and `fromJson`.
- `Layer`:
  - Added `toJson`, `toJsonMap` and `fromJson`.
- `ActivationFunction`:
  - Added `toJson`, `toJsonMap`, `fromJson` and `byName`.
- `Scale`:
  - Added `format`.
  - Added `toJson`, `toJsonMap` and `fromJson`.
- `Signal`:
  - Added `format` and `fromFormat`.
  - Optimize `values` implementation for each format.
- `Propagation` remove unused `_layersPreviousGradientsDeltas`.
- Extension `ListExtension`:
 - Added `asDoubles` and `asInts`.

## 1.1.2

- `ActivationFunctionSigmoid`:
  - Changed to use new faster `dart:math.exp` function.

## 1.1.1

- `ActivationFunction`:
  - Added base class `ActivationFunctionFloat32x4`.
  - SIMD Optimization:
    - Improved performance in 2x.
    - `ActivationFunctionLinear`, `ActivationFunctionSigmoid`,
      `ActivationFunctionSigmoidFast`, `ActivationFunctionSigmoidBoundedFast`.
- `eneural_net_fast_math.dart`:
  - `exp`: Improved performance and input range bounded to -87..87.
  - `expFloat32x4`: new SIMD Optimized Exponential function.
- `Chronometer`:
  - Improved `toString` numbers.
  - `Comparable`.
  - operator `+`.
- `eneural_net_extensions`:
  - Improved extensions.
  - Improved documentation.
- `Training`:
  - Added `logProgressEnabled`.
- intl: ^0.17.0

## 1.1.0

- `ActivationFunction`:
  - Added field `flatSpot` for `derivativeEntryWithFlatSpot()`.
  - Added `ActivationFunctionLinear`.
  - `ActivationFunctionSigmoid`: activation with bounds (-700 .. 700).
- Improved collections and numeric extensions.
- Improved `DataStatistics` and add `CSV` generator.
- `Signal`:
  - Added SIMD related operations.
  - Added: `computeSumSquaresMean`, `computeSumSquares`, `valuesAsDouble`.
  - Set extra values (out of length range): `setExtraValuesToZero`, `setExtraValuesToOne`, `setExtraValues`.
  - Improved documentation.
- `Sample`:
  - Input/Output statistics and proximity.
- Added `SamplesSet`:
  - With per set computed `defaultTargetGlobalError`.
  - Automatic `removeConflicts`.
- `Training`:
  - Split into `Propagation` and `ParameterStrategy`, allowing other algorithms.
  - Added `Backpropagation` with SIMD, smart learning rate and smart momentum.
  - Added `iRprop+`.
  - Added `TrainingLogger`.
  - Added `selectInitialANN`.
- `ANN`:
  - Optional bias neuron.
  - Allow different `ActivationFunction` for each layer.

## 1.0.2

- Expose fast math as an additional library.

## 1.0.1

- `README.md`:
  - Improve text.
  - Improve activation function text.
  - Fix example.

## 1.0.0

- Initial version.
- Training algorithms: Backpropagation.
- Activation functions: Sigmoid and approximation versions.
- Fast math functions.
- SIMD: Float32x4
