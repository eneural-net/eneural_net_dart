import 'dart:typed_data';

import 'eneural_net_fastmath.dart' as fast_math;

abstract class ActivationFunction<N extends num, E> {
  final int initialWeightScale;

  const ActivationFunction(this.initialWeightScale);

  N activate(N x);

  E activateEntry(E entry);

  N derivative(N o);

  E derivativeEntry(E entry);

  @override
  String toString() => runtimeType.toString();
}

class ActivationFunctionSigmoid extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionSigmoid([int initialWeightScale = 2])
      : super(initialWeightScale);

  @override
  double activate(double x) {
    return 1 / (1 + fast_math.exp(-x));
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    return Float32x4(
      1 / (1 + fast_math.exp(-entry.x)),
      1 / (1 + fast_math.exp(-entry.y)),
      1 / (1 + fast_math.exp(-entry.z)),
      1 / (1 + fast_math.exp(-entry.w)),
    );
  }

  @override
  double derivative(double o) {
    return o * (1.0 - o);
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
  }
}

class ActivationFunctionSigmoidFast
    extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionSigmoidFast([int initialWeightScale = 2])
      : super(initialWeightScale);

  @override
  double activate(double x) {
    x *= 3;
    return 0.5 + ((x) / (2.5 + x.abs()) / 2);
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    return Float32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
  }

  @override
  double derivative(double o) {
    return o * (1.0 - o);
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
  }
}

class ActivationFunctionSigmoidBoundedFast
    extends ActivationFunction<double, Float32x4> {
  final double lowerLimit;

  final double upperLimit;

  final double scale;

  const ActivationFunctionSigmoidBoundedFast(
      [this.scale = 6, int initialWeightScale = 2])
      : lowerLimit = -scale,
        upperLimit = scale,
        super(initialWeightScale);

  @override
  double activate(double x) {
    if (x < lowerLimit) {
      return 0.0;
    } else if (x > upperLimit) {
      return 1.0;
    }
    x = x / scale;
    return 0.5 + (x / (1 + (x * x)));
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    return Float32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
  }

  @override
  double derivative(double o) {
    return o * (1.0 - o);
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
  }
}

class ActivationFunctionSigmoidFastInt100
    extends ActivationFunction<int, Int32x4> {
  const ActivationFunctionSigmoidFastInt100([int initialWeightScale = 10])
      : super(initialWeightScale);

  @override
  int activate(int x) {
    return 50 + ((x * 100) ~/ (2 + x.abs()) ~/ 2);
  }

  @override
  Int32x4 activateEntry(Int32x4 entry) {
    return Int32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
  }

  @override
  int derivative(int o) {
    return o * (100 - o);
  }

  @override
  Int32x4 derivativeEntry(Int32x4 entry) {
    return Int32x4(
      entry.x * (100 - entry.x),
      entry.y * (100 - entry.y),
      entry.z * (100 - entry.z),
      entry.w * (100 - entry.w),
    );
  }
}

class ActivationFunctionSigmoidFastInt
    extends ActivationFunction<int, Int32x4> {
  final int scaleCenter;
  final int scaleMax;

  const ActivationFunctionSigmoidFastInt(this.scaleMax,
      [int initialWeightScale = 10])
      : scaleCenter = scaleMax ~/ 2,
        super(initialWeightScale);

  @override
  int activate(int x) {
    return scaleCenter + ((x * scaleMax) ~/ (2 + x.abs()) ~/ 2);
  }

  @override
  Int32x4 activateEntry(Int32x4 entry) {
    return Int32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
  }

  @override
  int derivative(int o) {
    return o * (scaleMax - o);
  }

  @override
  Int32x4 derivativeEntry(Int32x4 entry) {
    return Int32x4(
      entry.x * (100 - entry.x),
      entry.y * (100 - entry.y),
      entry.z * (100 - entry.z),
      entry.w * (100 - entry.w),
    );
  }
}
