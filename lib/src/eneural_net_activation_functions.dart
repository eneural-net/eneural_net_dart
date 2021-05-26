import 'dart:math';
import 'dart:typed_data';

import 'eneural_net_fastmath.dart' as fast_math;

enum ActivationFunctionScope { input, hidden, output, any }

abstract class ActivationFunction<N extends num, E> {
  static const double TOO_SMALL = -1.0E20;
  static const double TOO_BIG = 1.0E20;

  final double initialWeightScale;
  final double initialWeightBiasValue;
  final double flatSpot;

  const ActivationFunction(this.initialWeightScale,
      {this.initialWeightBiasValue = 0.0, this.flatSpot = 0.0001});

  List<ActivationFunctionScope> get scope;

  double createRandomWeight(Random random, {double? scale}) {
    scale ??= initialWeightScale;
    return (random.nextDouble() * (scale * 2)) - scale;
  }

  List<double> createRandomWeights(Random random, int length, {double? scale}) {
    return List.generate(length, (index) => createRandomWeight(random));
  }

  N activate(N x);

  E activateEntry(E entry);

  N derivative(N o);

  N derivativeWithFlatSpot(N o) {
    return derivative(o);
  }

  E derivativeEntry(E entry);

  E derivativeEntryWithFlatSpot(E entry) {
    return derivativeEntry(entry);
  }

  @override
  String toString() => runtimeType.toString();
}

class ActivationFunctionLinear extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionLinear({double initialWeightScale = 1})
      : super(initialWeightScale);

  static final List<ActivationFunctionScope> _scope =
      List.unmodifiable([ActivationFunctionScope.input]);

  @override
  List<ActivationFunctionScope> get scope => _scope;

  @override
  double activate(double x) {
    return x;
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    return entry;
  }

  @override
  double derivative(double o) {
    return 1.0;
  }

  @override
  double derivativeWithFlatSpot(double o) {
    return 1.0 + flatSpot;
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return Float32x4(
      1.0,
      1.0,
      1.0,
      1.0,
    );
  }

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return Float32x4(
      1.0 + flatSpot,
      1.0 + flatSpot,
      1.0 + flatSpot,
      1.0 + flatSpot,
    );
  }
}

class ActivationFunctionSigmoid extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionSigmoid({double initialWeightScale = 1})
      : super(initialWeightScale);

  static final List<ActivationFunctionScope> _scope = List.unmodifiable(
      [ActivationFunctionScope.hidden, ActivationFunctionScope.output]);

  @override
  List<ActivationFunctionScope> get scope => _scope;

  @override
  double activate(double x) {
    if (x < -700) {
      return 0.0;
    } else if (x > 700) {
      return 1.0;
    }

    return 1 / (1 + fast_math.exp(-x));
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
  double derivativeWithFlatSpot(double o) {
    return o * (1.0 - o) + flatSpot;
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

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return Float32x4(
      entry.x * (1.0 - entry.x) + flatSpot,
      entry.y * (1.0 - entry.y) + flatSpot,
      entry.z * (1.0 - entry.z) + flatSpot,
      entry.w * (1.0 - entry.w) + flatSpot,
    );
  }
}

class ActivationFunctionSigmoidFast
    extends ActivationFunction<double, Float32x4> {
  const ActivationFunctionSigmoidFast({double initialWeightScale = 1})
      : super(initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

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
      {this.scale = 6, double initialWeightScale = 2})
      : lowerLimit = -scale,
        upperLimit = scale,
        super(initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

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
  const ActivationFunctionSigmoidFastInt100([double initialWeightScale = 10])
      : super(initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

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
      [double initialWeightScale = 10])
      : scaleCenter = scaleMax ~/ 2,
        super(initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

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
