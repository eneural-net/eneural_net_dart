import 'dart:math' as math;
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;
typedef OptF =
    GradientOptimizer<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

/// Numeric unit tests of the per-weight update rule of each optimizer, exercised
/// directly via [GradientOptimizer.updateWeightEntry] with controlled inputs and
/// compared against the closed-form value of one step. (RProp already has such
/// tests in `eneural_net_rprop_test.dart`.)
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;
  final samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '1,1=0'],
    scale,
    true,
  );

  ANNF build() => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(2, true)],
    LayerFloat32x4(1, false),
    random: math.Random(1),
  );

  /// One update step for gradient [g] (with an optional previous gradient and a
  /// step count for Adam-style bias correction). Returns the delta added to the
  /// weight, per lane.
  double step(
    OptF opt,
    double g, {
    double previousGradient = 0.0,
    int optimizerStep = 1,
  }) {
    opt.optimizerStep = optimizerStep;
    final delta = opt.updateWeightEntry(
      layerIndex: 0,
      neuronIndex: 0,
      entryIndex: 0,
      weight: Float32x4.splat(0.3),
      gradient: Float32x4.splat(g),
      previousGradient: Float32x4.splat(previousGradient),
      neuronOutput: Float32x4.splat(1.0),
    );
    return delta.x;
  }

  test('SGD: delta = lr * g', () {
    final o = SGD(build(), SamplesSet(samples), learningRate: 0.1);
    expect(step(o, 0.5), closeTo(0.1 * 0.5, 1e-6));
    expect(step(o, -0.5), closeTo(0.1 * -0.5, 1e-6));
  });

  test('SGD+momentum: accumulates velocity', () {
    final o = SGD(
      build(),
      SamplesSet(samples),
      learningRate: 0.1,
      momentum: 0.9,
    );
    expect(step(o, 0.5), closeTo(0.1 * 0.5, 1e-6)); // v = g
    // second step (same g): v = 0.9*g + g = 1.9*g
    expect(step(o, 0.5), closeTo(0.1 * 1.9 * 0.5, 1e-6));
  });

  test('Adam: first step = lr * g / (|g| + eps)', () {
    const lr = 0.001, eps = 1e-8, g = 0.5;
    final o = Adam(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o, g), closeTo(lr * g / (g.abs() + eps), 1e-7));
    final o2 = Adam(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o2, -g), closeTo(lr * -g / (g.abs() + eps), 1e-7));
  });

  test('RMSProp: first step = lr * g / (sqrt((1-rho)*g^2) + eps)', () {
    const lr = 0.01, rho = 0.9, eps = 1e-8, g = 0.5;
    final o = RMSProp(build(), SamplesSet(samples), learningRate: lr, rho: rho);
    final expected = lr * g / (math.sqrt((1 - rho) * g * g) + eps);
    expect(step(o, g), closeTo(expected, 1e-6));
  });

  test('AdaGrad: first step = lr * g / (|g| + eps)', () {
    const lr = 0.05, eps = 1e-8, g = 0.5;
    final o = AdaGrad(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o, g), closeTo(lr * g / (math.sqrt(g * g) + eps), 1e-6));
  });

  test('AdaDelta: first step = sqrt(eps)/sqrt((1-rho)*g^2+eps) * g', () {
    const rho = 0.95, eps = 1e-6, g = 0.5;
    final o = AdaDelta(build(), SamplesSet(samples), rho: rho, epsilon: eps);
    final expected = math.sqrt(eps) / math.sqrt((1 - rho) * g * g + eps) * g;
    expect(step(o, g), closeTo(expected, 1e-6));
  });

  test('Lion: delta = lr * sign(momentum blend)', () {
    const lr = 0.02;
    final o = Lion(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o, 0.5), closeTo(lr, 1e-6)); // sign(+) = +1
    final o2 = Lion(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o2, -0.5), closeTo(-lr, 1e-6));
  });

  test('Quickprop: first step bootstraps with lr * g', () {
    const lr = 0.5, g = 0.4;
    final o = Quickprop(build(), SamplesSet(samples), learningRate: lr);
    expect(step(o, g), closeTo(lr * g, 1e-6));
  });

  test('ResilientPropagation (iRProp+): first step = sign(g) * stepInit', () {
    final o = ResilientPropagation(build(), SamplesSet(samples), stepInit: 0.1);
    // previousGradient = 0 -> change = 0 -> plain step of size stepInit.
    expect(step(o, 0.5), closeTo(0.1, 1e-6));
    final o2 = ResilientPropagation(
      build(),
      SamplesSet(samples),
      stepInit: 0.1,
    );
    expect(step(o2, -0.5), closeTo(-0.1, 1e-6));
  });

  test('ResilientPropagation: same-sign gradient accelerates by etaPlus', () {
    final o = ResilientPropagation(
      build(),
      SamplesSet(samples),
      stepInit: 0.1,
      etaPlus: 1.2,
    );
    step(o, 0.5); // sets step state; change 0 -> step stays 0.1
    // same sign again -> change > 0 -> step *= etaPlus = 0.12
    expect(step(o, 0.5, previousGradient: 0.5), closeTo(0.12, 1e-6));
  });
}
