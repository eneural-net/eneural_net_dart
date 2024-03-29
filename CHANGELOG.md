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
