import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Numerical gradient checking for the Backpropagation implementation.
///
/// The batch error minimized by the trainer is
/// `E(W) = Σ_samples Σ_outputs (target - output)²`.
///
/// For a single clean gradient-descent step with `momentum = 0` and
/// `learningRate = 1/N` (the defaults on the first epoch), the weights move by
///
///   ΔW = (1/N) · G     where G is the accumulated gradient the code computes.
///
/// So `G = ΔW · N` can be recovered from one `train(1, 0.0)` call, and compared
/// against the central-difference gradient of `E`. Because `E` carries no `½`
/// factor while the deltas use `(t-o)·f'(o)`, the analytical `G` equals
/// `-0.5 · ∂E/∂W`.
void main() {
  var scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  /// Central-difference gradient of the batch error at [w0].
  List<double> numericalGradient(
    ANNF ann,
    List<SampleFloat32x4> samples,
    List<double> w0, {
    double eps = 1e-2,
  }) {
    double batchError(List<double> w) {
      ann.allWeights = w;
      var e = 0.0;
      for (var s in samples) {
        ann.activate(s.input);
        var out = ann.output;
        var exp = s.output.values;
        for (var k = 0; k < out.length; k++) {
          var d = exp[k] - out[k];
          e += d * d;
        }
      }
      return e;
    }

    var g = List<double>.filled(w0.length, 0.0);
    for (var i = 0; i < w0.length; i++) {
      var wp = List<double>.from(w0)..[i] += eps;
      var wm = List<double>.from(w0)..[i] -= eps;
      g[i] = (batchError(wp) - batchError(wm)) / (2 * eps);
    }
    ann.allWeights = w0;
    return g;
  }

  /// The analytical gradient the trainer computes, recovered from one clean
  /// gradient-descent step.
  List<double> analyticalGradient(ANNF ann, List<SampleFloat32x4> samples) {
    var w0 = ann.allWeights;
    var bp = Backpropagation(ann, SamplesSet(samples, subject: 'grad'));
    bp.logEnabled = false;

    // First step: learningRate = 1/N, momentum = 0 (no momentum contribution).
    bp.train(1, 0.0);
    var w1 = ann.allWeights;

    var n = samples.length;
    var g = List<double>.generate(w0.length, (i) => (w1[i] - w0[i]) * n);

    ann.allWeights = w0; // restore
    return g;
  }

  double cosine(List<double> a, List<double> b) {
    var dot = 0.0, na = 0.0, nb = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (sqrt(na) * sqrt(nb));
  }

  group('Backpropagation gradient check', () {
    test(
      'analytical gradient matches the numerical gradient (2 -> 3 -> 1)',
      () {
        var samples = xor();
        var ann = ANN(
          scale,
          LayerFloat32x4(2, true),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false),
          random: Random(101),
        );

        var w0 = ann.allWeights;
        var numG = numericalGradient(ann, samples, w0);
        var anaG = analyticalGradient(ann, samples);

        // Expected relationship: analytical = -0.5 * numerical.
        var pred = numG.map((g) => -0.5 * g).toList();

        expect(
          cosine(anaG, pred),
          closeTo(1.0, 1e-3),
          reason: 'gradient direction must match the numerical gradient',
        );

        // Every weight (including the bias weights) must match within the
        // tolerance allowed by float32 precision and the flat-spot term.
        for (var i = 0; i < anaG.length; i++) {
          var rel = (anaG[i] - pred[i]).abs() / (pred[i].abs() + 1e-3);
          expect(
            rel < 0.05,
            isTrue,
            reason:
                'weight $i: analytical ${anaG[i]} vs expected ${pred[i]} '
                '(rel err $rel)',
          );
        }
      },
    );

    test('gradient check for a deeper network (2 -> 4 -> 3 -> 1)', () {
      var samples = xor();
      var ann = ANN(
        scale,
        LayerFloat32x4(2, true),
        [HiddenLayerConfig(4, true), HiddenLayerConfig(3, true)],
        LayerFloat32x4(1, false),
        random: Random(7),
      );

      var w0 = ann.allWeights;
      var pred = numericalGradient(
        ann,
        samples,
        w0,
      ).map((g) => -0.5 * g).toList();
      var anaG = analyticalGradient(ann, samples);

      expect(cosine(anaG, pred), closeTo(1.0, 1e-3));
    });

    test('gradient check without bias neurons', () {
      var samples = xor();
      var ann = ANN(
        scale,
        LayerFloat32x4(2, false),
        [HiddenLayerConfig(4, false)],
        LayerFloat32x4(1, false),
        random: Random(3),
      );

      var w0 = ann.allWeights;
      var pred = numericalGradient(
        ann,
        samples,
        w0,
      ).map((g) => -0.5 * g).toList();
      var anaG = analyticalGradient(ann, samples);

      expect(cosine(anaG, pred), closeTo(1.0, 1e-3));
    });

    test('gradient check with the SigmoidFast activation', () {
      var samples = xor();
      var ann = ANN(
        scale,
        LayerFloat32x4(2, true, ActivationFunctionSigmoidFast()),
        [HiddenLayerConfig(3, true)],
        LayerFloat32x4(1, false, ActivationFunctionSigmoidFast()),
        random: Random(5),
      );

      var w0 = ann.allWeights;
      var pred = numericalGradient(
        ann,
        samples,
        w0,
      ).map((g) => -0.5 * g).toList();
      var anaG = analyticalGradient(ann, samples);

      // SigmoidFast reuses the plain sigmoid derivative `o(1-o)`, which does
      // NOT match the true derivative of its pseudo-sigmoid activation, so the
      // gradient is only an approximate descent direction (cosine ~0.85). It
      // still points downhill, which is why the "Fast" family trains, but this
      // is why it converges less precisely than the exact `Sigmoid`.
      expect(cosine(anaG, pred), greaterThan(0.8));
    });
  });

  group('Backpropagation: bias neurons learn', () {
    test('input-layer bias weights receive a non-zero gradient', () {
      // Regression test: the input bias propagates a constant 1 in the forward
      // pass, so its outgoing weights must get a real gradient (they used to be
      // frozen at their initial value because the stored bias slot held 0).
      var samples = xor();
      var ann = ANN(
        scale,
        LayerFloat32x4(2, true),
        [HiddenLayerConfig(3, true)],
        LayerFloat32x4(1, false),
        random: Random(101),
      );

      // Input-bias -> hidden weights are the last block of `allWeights`:
      // hidden(4)->out (indices 0..3) then input(3)->hidden(4) (indices 4..15),
      // with input neuron 2 (the bias) at indices 12..15.
      var biasStart = 12;
      var before = ann.allWeights.sublist(biasStart, biasStart + 4);
      expect(
        before.every((w) => w == 0.0),
        isTrue,
        reason: 'bias weights start at the neutral initial value',
      );

      var training = Backpropagation(ann, SamplesSet(samples, subject: 'xor'));
      training.logEnabled = false;
      training.enableSelectInitialANN = false;
      training.train(500, 0.0);

      var after = ann.allWeights.sublist(biasStart, biasStart + 4);

      // At least one real input-bias weight must have moved (the 4th connects
      // to the hidden bias slot and legitimately stays ~0).
      expect(
        after.take(3).any((w) => w.abs() > 0.1),
        isTrue,
        reason: 'input-bias weights must learn, not stay frozen at 0',
      );
    });

    test('a bias-driven mapping is learnable (constant target)', () {
      // Output must be ~0.9 regardless of input — only achievable through the
      // bias, since the inputs carry no information about the target.
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0.9', '0,1=0.9', '1,0=0.9', '1,1=0.9'],
        scale,
        true,
      );

      var ann = ANN(
        scale,
        LayerFloat32x4(2, true),
        [HiddenLayerConfig(3, true)],
        LayerFloat32x4(1, false),
        random: Random(11),
      );

      var training = Backpropagation(
        ann,
        SamplesSet(samples, subject: 'const'),
      );
      training.logEnabled = false;
      training.enableSelectInitialANN = false;
      training.train(3000, 0.0);

      for (var s in samples) {
        ann.activate(s.input);
        expect(ann.output.first, closeTo(0.9, 0.05));
      }
    });
  });
}
