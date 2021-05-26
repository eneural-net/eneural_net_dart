## 1.0.3

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
