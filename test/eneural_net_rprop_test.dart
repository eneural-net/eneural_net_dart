import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

typedef RPropF =
    RProp<double, Float32x4, SignalFloat32x4, Scale<double>, SampleFloat32x4>;

/// Correctness tests for the Resilient Backpropagation (iRProp+) trainer.
///
/// iRProp+ (Igel & Hüsken, 2000) adapts a per-weight step size using only the
/// SIGN of the batch gradient:
///
///  - sign unchanged  -> step *= η⁺ (1.2), capped at Δmax
///  - sign flipped     -> step *= η⁻ (0.5), floored at Δmin; the last weight
///                         step is reverted iff the total error increased, and
///                         the next iteration is forced to make a plain step
///                         (no further acceleration/deceleration)
///  - sign is zero     -> plain step with the unchanged step size
void main() {
  var scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({int hidden = 3, int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(hidden, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  /// A trainer with `trainingSamplesSize` initialized so `computeWeightUpdate`
  /// can be exercised directly.
  RPropF readyTrainer() {
    var t = RProp(build(), SamplesSet(xor(), subject: 'xor'));
    t.logEnabled = false;
    t.train(1, 0.0);
    return t;
  }

  group('RProp: only the gradient SIGN matters (magnitude independence)', () {
    test('same sign, wildly different magnitudes -> identical update', () {
      var t = readyTrainer();

      List<double> step(double gradient, double previousGradient) {
        var deltas = <num>[0.1, 0.1, 0.1, 0.1];
        var counters = <num>[0.0, 0.0, 0.0, 0.0];
        var wu = t.computeWeightUpdate(
          0.5,
          0.0,
          gradient,
          previousGradient,
          deltas,
          counters,
          0,
          0.0,
        );
        return [wu, deltas[0].toDouble()];
      }

      var tiny = step(1e-4, 2e-4);
      var huge = step(9999.0, 5000.0);

      expect(tiny[0], equals(huge[0]), reason: 'weight update must ignore |g|');
      expect(tiny[1], equals(huge[1]), reason: 'step size must ignore |g|');
      expect(tiny[1], closeTo(0.12, 1e-12), reason: 'grew by η⁺ = 1.2');
    });

    test('opposite signs of the same magnitude -> opposite update', () {
      var t = readyTrainer();

      double step(double gradient) {
        var deltas = <num>[0.1, 0.1, 0.1, 0.1];
        var counters = <num>[0.0, 0.0, 0.0, 0.0];
        return t.computeWeightUpdate(
          0.5,
          0.0,
          gradient,
          gradient, // same-sign history -> plain accelerated step
          deltas,
          counters,
          0,
          0.0,
        );
      }

      expect(step(3.0), equals(-step(-3.0)));
      expect(step(3.0) > 0, isTrue, reason: 'descends along +gradient');
    });
  });

  group('RProp: step-size adaptation', () {
    test('unchanged sign accelerates by η⁺ = 1.2', () {
      var t = readyTrainer();
      var deltas = <num>[0.1, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      var wu = t.computeWeightUpdate(
        0.5,
        0.0,
        1.0,
        1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(deltas[0], closeTo(0.12, 1e-12));
      expect(
        wu,
        closeTo(0.12, 1e-12),
        reason: 'step in the gradient direction',
      );
    });

    test('flipped sign decelerates by η⁻ = 0.5 and sets the reversal flag', () {
      var t = readyTrainer();
      var deltas = <num>[0.1, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      var wu = t.computeWeightUpdate(
        0.5,
        0.3,
        1.0,
        -1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      // Stored as negative: the magnitude is the halved step, the sign is the
      // "do not change direction next iteration" flag.
      expect(deltas[0], closeTo(-0.05, 1e-12));
      // Error did not increase in this fresh trainer -> no step this epoch.
      expect(wu, equals(0.0));
    });

    test('the reversal flag makes the next iteration a plain step', () {
      var t = readyTrainer();

      // previousUpdateDeltas came in NEGATIVE (flag set by a prior sign flip).
      var deltas = <num>[-0.05, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      var wu = t.computeWeightUpdate(
        0.5,
        0.0,
        2.0,
        8.0, // same sign -> would normally accelerate...
        deltas,
        counters,
        0,
        0.0,
      );

      // ...but the flag forces a plain step: restored to +0.05, NOT 0.06.
      expect(deltas[0], closeTo(0.05, 1e-12), reason: 'restored to positive');
      expect(wu, closeTo(0.05, 1e-12), reason: 'plain step, not accelerated');
    });

    test('step size is capped at Δmax = 50', () {
      var t = readyTrainer();
      var deltas = <num>[100.0, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      t.computeWeightUpdate(0.5, 0.0, 1.0, 1.0, deltas, counters, 0, 0.0);

      expect(deltas[0], equals(RProp.weightMaxStep));
      expect(RProp.weightMaxStep, equals(50.0));
    });

    test('step size is floored at Δmin = 1e-6', () {
      var t = readyTrainer();
      var deltas = <num>[1e-9, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      t.computeWeightUpdate(0.5, 0.0, 1.0, -1.0, deltas, counters, 0, 0.0);

      // Floored then stored with the reversal-flag sign.
      expect(deltas[0], closeTo(-RProp.weightMinStep, 1e-18));
      expect(RProp.weightMinStep, equals(1.0e-6));
    });

    test('a zero gradient makes no step and keeps the step size', () {
      var t = readyTrainer();
      var deltas = <num>[0.1, 0, 0, 0];
      var counters = <num>[0.0, 0, 0, 0];

      var wu = t.computeWeightUpdate(
        0.5,
        0.0,
        0.0,
        1.0,
        deltas,
        counters,
        0,
        0.0,
      );

      expect(wu, equals(0.0));
      expect(deltas[0], closeTo(0.1, 1e-12));
    });
  });

  group('RProp: convergence', () {
    test('drives XOR to machine precision', () {
      var ann = build();
      var t = RProp(ann, SamplesSet(xor(), subject: 'xor'));
      t.logEnabled = false;

      t.train(600, 0.0);

      expect(
        ann.computeSamplesGlobalError(xor()),
        lessThan(1e-6),
        reason: 'RProp should reach a very low error',
      );
    });

    test('backtracking keeps the error from rising above its start', () {
      // iRProp+ reverts sign-flipped steps when the total error increases, so
      // the error never climbs above where it began and any transient bump is
      // tiny — but it is NOT strictly monotonic (unreverted weights still move).
      for (var seed in [3, 7, 101]) {
        var ann = build(seed: seed);
        var t = RProp(ann, SamplesSet(xor(), subject: 'xor'));
        t.logEnabled = false;

        var initial = ann.computeSamplesGlobalError(xor());
        var maxErr = initial;
        var worstJump = 0.0;
        var previous = initial;

        for (var i = 0; i < 40; i++) {
          t.train(10, 0.0);
          var err = ann.computeSamplesGlobalError(xor());
          if (err > maxErr) maxErr = err;
          if (err - previous > worstJump) worstJump = err - previous;
          previous = err;
        }

        expect(
          maxErr,
          lessThanOrEqualTo(initial * 1.001),
          reason: 'seed $seed: error must never exceed its starting value',
        );
        expect(
          worstJump,
          lessThan(1e-3),
          reason: 'seed $seed: transient increases must stay tiny',
        );
      }
    });

    test('converges markedly faster than plain batch backpropagation', () {
      var samples = xor();

      var annBp = build(seed: 7);
      var bp = Backpropagation(annBp, SamplesSet(samples));
      bp.logEnabled = false;
      bp.enableSelectInitialANN = false;
      bp.train(400, 0.0);

      var annRp = build(seed: 7);
      var rp = RProp(annRp, SamplesSet(samples));
      rp.logEnabled = false;
      rp.train(400, 0.0);

      var errBp = annBp.computeSamplesGlobalError(samples);
      var errRp = annRp.computeSamplesGlobalError(samples);

      expect(
        errRp < errBp,
        isTrue,
        reason: 'RProp ($errRp) should beat batch backprop ($errBp)',
      );
      expect(errRp, lessThan(1e-3));
    });

    test('the first step uses the initial step size Δ₀ = 0.1', () {
      var ann = build();
      var w0 = ann.allWeights;

      var t = RProp(ann, SamplesSet(xor(), subject: 'xor'));
      t.logEnabled = false;
      t.train(1, 0.0); // one epoch: first RProp step

      var w1 = ann.allWeights;

      // Every weight that received a non-zero gradient moves by exactly ±0.1.
      var moved = <double>[];
      for (var i = 0; i < w0.length; i++) {
        var d = (w1[i] - w0[i]).abs();
        if (d > 1e-9) moved.add(d);
      }

      expect(moved, isNotEmpty);
      for (var d in moved) {
        expect(d, closeTo(0.1, 1e-4), reason: 'first step magnitude is Δ₀');
      }
    });
  });

  group('RProp: robustness', () {
    test('trains reliably across many seeds', () {
      var samples = xor();
      var converged = 0;

      for (var seed = 0; seed < 12; seed++) {
        var ann = build(hidden: 4, seed: seed);
        var t = RProp(ann, SamplesSet(samples, subject: 'xor'));
        t.logEnabled = false;
        t.trainUntilGlobalError(
          targetGlobalError: 0.01,
          maxRetries: 15,
          random: Random(seed),
        );
        if (ann.computeSamplesGlobalError(samples) <= 0.01) converged++;
      }

      expect(
        converged,
        greaterThanOrEqualTo(11),
        reason: 'RProp should converge on XOR for almost every seed',
      );
    });

    test('learns a bias-only constant mapping', () {
      // Output must be ~0.85 regardless of input -> only the bias can encode it.
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0.85', '0,1=0.85', '1,0=0.85', '1,1=0.85'],
        scale,
        true,
      );

      var ann = build(seed: 5);
      var t = RProp(ann, SamplesSet(samples, subject: 'const'));
      t.logEnabled = false;
      t.train(1000, 0.0);

      for (var s in samples) {
        ann.activate(s.input);
        expect(ann.output.first, closeTo(0.85, 0.03));
      }
    });
  });
}
