import 'dart:typed_data';

import 'eneural_net_fastmath.dart' as fast_math;

abstract class ActivationFunction<N extends num, E> {
  final int initialWeightScale;

  const ActivationFunction(this.initialWeightScale);

  N activate(N n);

  E activateX4(E entry);

  N derivative(N o);

  E derivativeX4(E entry);

  @override
  String toString() => runtimeType.toString();
}

class ActivationFunctionSigmoid extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionSigmoid([int initialWeightScale = 2])
      : super(initialWeightScale);

  @override
  double activate(double n) {
    return 1 / (1 + fast_math.exp(-n));
  }

  @override
  Float32x4 activateX4(Float32x4 entry) {
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
  Float32x4 derivativeX4(Float32x4 entry) {
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
  double activate(double n) {
    n *= 3;
    return 0.5 + ((n) / (2.5 + n.abs()) / 2);
  }

  @override
  Float32x4 activateX4(Float32x4 entry) {
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
  Float32x4 derivativeX4(Float32x4 entry) {
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
  double activate(double n) {
    if (n < lowerLimit) {
      return 0.0;
    } else if (n > upperLimit) {
      return 1.0;
    }
    n = n / scale;
    return 0.5 + (n / (1 + (n * n)));
  }

  @override
  Float32x4 activateX4(Float32x4 entry) {
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
  Float32x4 derivativeX4(Float32x4 entry) {
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
  int activate(int n) {
    return 50 + ((n * 100) ~/ (2 + n.abs()) ~/ 2);
  }

  @override
  Int32x4 activateX4(Int32x4 entry) {
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
  Int32x4 derivativeX4(Int32x4 entry) {
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
  int activate(int n) {
    return scaleCenter + ((n * scaleMax) ~/ (2 + n.abs()) ~/ 2);
  }

  @override
  Int32x4 activateX4(Int32x4 entry) {
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
  Int32x4 derivativeX4(Int32x4 entry) {
    return Int32x4(
      entry.x * (100 - entry.x),
      entry.y * (100 - entry.y),
      entry.z * (100 - entry.z),
      entry.w * (100 - entry.w),
    );
  }
}
