import 'dart:math';
import 'dart:typed_data';

import 'package:swiss_knife/swiss_knife.dart';

//import 'eneural_net_fastmath.dart' as fast_math;

/// Scope of the activation function.
enum ActivationFunctionScope {
  /// Input layers.
  input,

  /// Hidden layers.
  hidden,

  /// Output layers.
  output,

  /// Any layer.
  any,
}

/// Base class for Activation Functions.
abstract class ActivationFunction<N extends num, E> {
  static ActivationFunction<N, E> byName<N extends num, E>(
    String name, {
    double? initialWeightScale,
    double? scale,
  }) {
    switch (name) {
      case 'Linear':
        return ActivationFunctionLinear(
                initialWeightScale: initialWeightScale ?? 1.0)
            as ActivationFunction<N, E>;
      case 'Sigmoid':
        return ActivationFunctionSigmoid(
                initialWeightScale: initialWeightScale ?? 1.0)
            as ActivationFunction<N, E>;
      case 'SigmoidFast':
        return ActivationFunctionSigmoidFast(
                initialWeightScale: initialWeightScale ?? 1.0)
            as ActivationFunction<N, E>;
      case 'SigmoidBoundedFast':
        return ActivationFunctionSigmoidBoundedFast(
            initialWeightScale: initialWeightScale ?? 1.0,
            scale: scale ?? 6.0) as ActivationFunction<N, E>;
      default:
        throw StateError('Unknown ActivationFunction with name: $name');
    }
  }

  /// Decodes JSON.
  static ActivationFunction<N, E> fromJson<N extends num, E>(dynamic json) {
    Map<String, dynamic> jsonMap = json is String ? parseJSON(json) : json;

    return byName(jsonMap['name'],
        initialWeightScale: jsonMap['initialWeightScale'],
        scale: jsonMap['scale']);
  }

  static const double TOO_SMALL = -1.0E20;
  static const double TOO_BIG = 1.0E20;

  final String name;
  final double initialWeightScale;
  final double initialWeightBiasValue;
  final double flatSpot;

  const ActivationFunction(this.name, this.initialWeightScale,
      {this.initialWeightBiasValue = 0.0, this.flatSpot = 0.0001});

  /// Scopes where this activation function should be used.
  List<ActivationFunctionScope> get scope;

  /// Generates a random weight compatible with this activation function.
  double createRandomWeight(Random random, {double? scale}) {
    scale ??= initialWeightScale;
    return (random.nextDouble() * (scale * 2)) - scale;
  }

  /// Generates a [List] of random weights compatible with this activation function.
  List<double> createRandomWeights(Random random, int length, {double? scale}) {
    return List.generate(length, (index) => createRandomWeight(random));
  }

  /// The activation function.
  N activate(N x);

  /// The activation function for an entry (SIMD).
  E activateEntry(E entry);

  /// The derivative function.
  N derivative(N o);

  /// The derivative function with `flat spot`.
  N derivativeWithFlatSpot(N o) {
    return derivative(o);
  }

  /// The derivative function for an entry (SIMD).
  E derivativeEntry(E entry);

  /// The derivative function for an entry with `flat spot` (SIMD).
  E derivativeEntryWithFlatSpot(E entry) {
    return derivativeEntry(entry);
  }

  @override
  String toString() => runtimeType.toString();

  /// Converts to an encoded JSON.
  String toJson({bool withIndent = false}) =>
      encodeJSON(toJsonMap(), withIndent: withIndent);

  /// Converts to a JSON [Map].
  Map<String, dynamic> toJsonMap() =>
      <String, dynamic>{'name': name, 'initialWeightScale': initialWeightScale};
}

/// Base class for SIMD optimized functions using [Float32x4].
abstract class ActivationFunctionFloat32x4
    extends ActivationFunction<double, Float32x4> {
  static final Float32x4 entryOfZeroes = Float32x4.splat(0.0);
  static final Float32x4 entryOfOnes = Float32x4.splat(1.0);
  static final Float32x4 entryOfMinusOnes = Float32x4.splat(-1.0);
  static final Float32x4 entryOfHalf = Float32x4.splat(0.50);
  static final Float32x4 entryOfTwos = Float32x4.splat(2.0);
  static final Float32x4 entryOfTwosAndHalf = Float32x4.splat(2.5);
  static final Float32x4 entryOfThrees = Float32x4.splat(3.0);

  final Float32x4 entryFlatSpot;

  ActivationFunctionFloat32x4(String name, double initialWeightScale,
      {double initialWeightBiasValue = 0.0, double flatSpot = 0.0001})
      : entryFlatSpot = Float32x4(flatSpot, flatSpot, flatSpot, flatSpot),
        super(name, initialWeightScale,
            initialWeightBiasValue: initialWeightBiasValue, flatSpot: flatSpot);
}

/// Linear Activation Function (SIMD optimized).
class ActivationFunctionLinear extends ActivationFunctionFloat32x4 {
  late final Float32x4 entryFlatSpotPlusOne;

  ActivationFunctionLinear({double initialWeightScale = 1})
      : super('Linear', initialWeightScale) {
    entryFlatSpotPlusOne =
        entryFlatSpot + ActivationFunctionFloat32x4.entryOfOnes;
  }

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
    return ActivationFunctionFloat32x4.entryOfOnes;
    /*
    return Float32x4(
      1.0,
      1.0,
      1.0,
      1.0,
    );
     */
  }

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return entryFlatSpotPlusOne;
    /*
    return Float32x4(
      1.0 + flatSpot,
      1.0 + flatSpot,
      1.0 + flatSpot,
      1.0 + flatSpot,
    );
     */
  }
}

/// Sigmoid Activation Function (SIMD optimized).
class ActivationFunctionSigmoid extends ActivationFunctionFloat32x4 {
  ActivationFunctionSigmoid({double initialWeightScale = 1})
      : super('Sigmoid', initialWeightScale);

  static final List<ActivationFunctionScope> _scope = List.unmodifiable(
      [ActivationFunctionScope.hidden, ActivationFunctionScope.output]);

  @override
  List<ActivationFunctionScope> get scope => _scope;

  @override
  double activate(double x) {
    return 1 / (1 + exp(-x));
    //return 1 / (1 + fast_math.exp(-x));
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    // New Dart v2.13.1 implementation of `exp` is very fast:
    var exp32x4 = Float32x4(
      exp(-entry.x),
      exp(-entry.y),
      exp(-entry.z),
      exp(-entry.w),
    );

    return ActivationFunctionFloat32x4.entryOfOnes /
        (ActivationFunctionFloat32x4.entryOfOnes + exp32x4);

    /*
    // SIMD version with `fast_math.expFloat32x4`
    return ActivationFunctionFloat32x4.entryOfOnes /
        (ActivationFunctionFloat32x4.entryOfOnes +
            fast_math.expFloat32x4(-entry));

     */

    /*
    // Non-SIMD version:
    return Float32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
     */
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
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry);
    /*
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
     */
  }

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry) +
        entryFlatSpot;
    /*
    return Float32x4(
      entry.x * (1.0 - entry.x) + flatSpot,
      entry.y * (1.0 - entry.y) + flatSpot,
      entry.z * (1.0 - entry.z) + flatSpot,
      entry.w * (1.0 - entry.w) + flatSpot,
    );
     */
  }
}

/// Fast Pseudo-Sigmoid Activation Function (SIMD optimized).
class ActivationFunctionSigmoidFast extends ActivationFunctionFloat32x4 {
  ActivationFunctionSigmoidFast({double initialWeightScale = 1})
      : super('SigmoidFast', initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

  @override
  double activate(double x) {
    x *= 3;
    return 0.5 + ((x) / (2.5 + x.abs()) / 2);
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    entry = entry * ActivationFunctionFloat32x4.entryOfThrees;
    return ActivationFunctionFloat32x4.entryOfHalf +
        ((entry) /
            (ActivationFunctionFloat32x4.entryOfTwosAndHalf + entry.abs()) /
            ActivationFunctionFloat32x4.entryOfTwos);
    /*
    return Float32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
     */
  }

  @override
  double derivative(double o) {
    return o * (1.0 - o);
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry);
    /*
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
     */
  }

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry) +
        entryFlatSpot;
  }
}

/// Fast Pseudo-Sigmoid Activation Function Bounded (SIMD optimized).
class ActivationFunctionSigmoidBoundedFast extends ActivationFunctionFloat32x4 {
  final double lowerLimit;

  final Float32x4 _entryLowerLimit;

  final double upperLimit;

  final Float32x4 _entryUpperLimit;

  final double scale;

  ActivationFunctionSigmoidBoundedFast(
      {this.scale = 6, double initialWeightScale = 2})
      : lowerLimit = -scale,
        _entryLowerLimit = Float32x4.splat(-scale),
        upperLimit = scale,
        _entryUpperLimit = Float32x4.splat(scale),
        super('SigmoidBoundedFast', initialWeightScale);

  @override
  List<ActivationFunctionScope> get scope => ActivationFunctionSigmoid._scope;

  @override
  double activate(double x) {
    x = x.clamp(lowerLimit, upperLimit);
    x = x / scale;
    return 0.5 + (x / (1 + (x * x)));
  }

  @override
  Float32x4 activateEntry(Float32x4 entry) {
    entry = entry.clamp(_entryLowerLimit, _entryUpperLimit);
    entry = entry * ActivationFunctionFloat32x4.entryOfThrees;
    return ActivationFunctionFloat32x4.entryOfHalf +
        ((entry) /
            (ActivationFunctionFloat32x4.entryOfTwosAndHalf + entry.abs()) /
            ActivationFunctionFloat32x4.entryOfTwos);
    /*
    return Float32x4(
      activate(entry.x),
      activate(entry.y),
      activate(entry.z),
      activate(entry.w),
    );
     */
  }

  @override
  double derivative(double o) {
    return o * (1.0 - o);
  }

  @override
  Float32x4 derivativeEntry(Float32x4 entry) {
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry);
    /*
    return Float32x4(
      entry.x * (1.0 - entry.x),
      entry.y * (1.0 - entry.y),
      entry.z * (1.0 - entry.z),
      entry.w * (1.0 - entry.w),
    );
     */
  }

  @override
  Float32x4 derivativeEntryWithFlatSpot(Float32x4 entry) {
    return entry * (ActivationFunctionFloat32x4.entryOfOnes - entry) +
        entryFlatSpot;
  }

  @override
  Map<String, dynamic> toJsonMap() {
    var json = super.toJsonMap();
    json['scale'] = scale;
    return json;
  }
}

/// Experimental Integer Sigmoid Function (scale 100).
class ActivationFunctionSigmoidFastInt100
    extends ActivationFunction<int, Int32x4> {
  const ActivationFunctionSigmoidFastInt100([double initialWeightScale = 10])
      : super('SigmoidFastInt100', initialWeightScale);

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

/// Experimental Integer Sigmoid Function.
class ActivationFunctionSigmoidFastInt
    extends ActivationFunction<int, Int32x4> {
  final int scaleCenter;
  final int scaleMax;

  const ActivationFunctionSigmoidFastInt(this.scaleMax,
      [double initialWeightScale = 10])
      : scaleCenter = scaleMax ~/ 2,
        super('SigmoidFastInt', initialWeightScale);

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
