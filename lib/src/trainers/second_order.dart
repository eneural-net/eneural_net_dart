import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

import '../eneural_net_linalg.dart';

/// Sample type accepted by the vector/second-order trainers (Float32x4 only).
typedef VectorSample =
    Sample<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Base for trainers that operate on the flat weight vector ([ANN.allWeights])
/// and use finite-difference gradients/Jacobians — robust for the small-network
/// regime these second-order methods target.
abstract class VectorTrainer<P extends VectorSample>
    extends Training<double, Float32x4, SignalFloat32x4, Scale<double>, P> {
  /// Finite-difference step.
  final double fdEpsilon;

  VectorTrainer(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet,
    String algorithmName, {
    this.fdEpsilon = 1e-4,
    String? subject,
  }) : super(ann, samplesSet, algorithmName, subject: subject);

  @override
  String get parameters => 'fdEpsilon: $fdEpsilon';

  List<double> get weights => ann.allWeights;
  set weights(List<double> w) => ann.allWeights = w;

  /// Scalar objective (mean squared error over all samples).
  double objective() => ann.computeSamplesGlobalError(samples);

  /// Central-difference gradient of [objective] w.r.t. every weight.
  List<double> gradient() {
    final w = weights;
    final n = w.length;
    final g = List<double>.filled(n, 0);
    for (var i = 0; i < n; ++i) {
      final orig = w[i];
      w[i] = orig + fdEpsilon;
      weights = w;
      final fPlus = objective();
      w[i] = orig - fdEpsilon;
      weights = w;
      final fMinus = objective();
      w[i] = orig;
      g[i] = (fPlus - fMinus) / (2 * fdEpsilon);
    }
    weights = w; // restore
    return g;
  }

  /// Per-sample per-output residuals `output − target` (for Levenberg–Marquardt).
  List<double> residuals() {
    final r = <double>[];
    for (final s in samples) {
      ann.activate(s.input);
      final out = ann.outputAsDouble;
      final target = s.output.valuesAsDouble;
      for (var k = 0; k < out.length; ++k) {
        r.add(out[k] - target[k]);
      }
    }
    return r;
  }

  /// Central-difference Jacobian of [residuals] (rows = residuals, cols = weights).
  List<List<double>> jacobian() {
    final w = weights;
    final n = w.length;
    final rows = residuals().length;
    final j = List.generate(rows, (_) => List<double>.filled(n, 0));
    for (var i = 0; i < n; ++i) {
      final orig = w[i];
      w[i] = orig + fdEpsilon;
      weights = w;
      final rp = residuals();
      w[i] = orig - fdEpsilon;
      weights = w;
      final rm = residuals();
      w[i] = orig;
      for (var k = 0; k < rows; ++k) {
        j[k][i] = (rp[k] - rm[k]) / (2 * fdEpsilon);
      }
    }
    weights = w;
    return j;
  }

  /// Backtracking (Armijo) line search along [direction] from [base]; returns
  /// the step size. Leaves the weights set to `base` on exit.
  double lineSearch(
    List<double> base,
    List<double> direction,
    double f0,
    List<double> grad, {
    double alpha0 = 1.0,
  }) {
    final gd = dot(grad, direction);
    const c = 1e-4;
    const shrink = 0.5;
    var alpha = alpha0;
    var best = alpha;
    for (var it = 0; it < 30; ++it) {
      final trial = List.generate(
        base.length,
        (i) => base[i] + alpha * direction[i],
      );
      weights = trial;
      final f = objective();
      if (f <= f0 + c * alpha * gd) {
        weights = base;
        return alpha;
      }
      best = alpha;
      alpha *= shrink;
    }
    weights = base;
    return best;
  }
}

/// Nonlinear Conjugate Gradient (Fletcher–Reeves, with restarts + line search).
class ConjugateGradient<P extends VectorSample> extends VectorTrainer<P> {
  List<double>? _prevGrad;
  List<double>? _prevDir;
  int _iter = 0;

  ConjugateGradient(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    double fdEpsilon = 1e-4,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'ConjugateGradient',
         fdEpsilon: fdEpsilon,
         subject: subject,
       );

  @override
  void reset() {
    super.reset();
    _prevGrad = null;
    _prevDir = null;
    _iter = 0;
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final w = weights;
    final g = gradient();
    final n = g.length;

    List<double> d;
    if (_prevGrad == null || _iter % n == 0) {
      d = g.map((x) => -x).toList();
    } else {
      final beta = max(
        0.0,
        dot(g, g) / max(dot(_prevGrad!, _prevGrad!), 1e-30),
      );
      d = List.generate(n, (i) => -g[i] + beta * _prevDir![i]);
      if (dot(g, d) > 0) d = g.map((x) => -x).toList();
    }

    final f0 = objective();
    final alpha = lineSearch(w, d, f0, g);
    weights = List.generate(n, (i) => w[i] + alpha * d[i]);

    _prevGrad = g;
    _prevDir = d;
    _iter++;
    return objective() <= targetGlobalError;
  }
}

/// Limited-memory BFGS (two-loop recursion + line search).
class LBFGS<P extends VectorSample> extends VectorTrainer<P> {
  final int memory;
  final List<List<double>> _s = [];
  final List<List<double>> _y = [];
  List<double>? _lastGrad;

  LBFGS(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.memory = 10,
    double fdEpsilon = 1e-4,
    String? subject,
  }) : super(ann, samplesSet, 'L-BFGS', fdEpsilon: fdEpsilon, subject: subject);

  @override
  void reset() {
    super.reset();
    _s.clear();
    _y.clear();
    _lastGrad = null;
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final w = weights;
    final g = _lastGrad ?? gradient();
    final n = g.length;
    final k = _s.length;

    // Two-loop recursion for d = -H·g.
    final q = List<double>.of(g);
    final alphas = List<double>.filled(k, 0);
    final rho = List<double>.filled(k, 0);
    for (var i = k - 1; i >= 0; --i) {
      rho[i] = 1.0 / max(dot(_y[i], _s[i]), 1e-30);
      alphas[i] = rho[i] * dot(_s[i], q);
      for (var j = 0; j < n; ++j) {
        q[j] -= alphas[i] * _y[i][j];
      }
    }
    var gamma = 1.0;
    if (k > 0) {
      gamma = dot(_s[k - 1], _y[k - 1]) / max(dot(_y[k - 1], _y[k - 1]), 1e-30);
    }
    for (var j = 0; j < n; ++j) {
      q[j] *= gamma;
    }
    for (var i = 0; i < k; ++i) {
      final beta = rho[i] * dot(_y[i], q);
      for (var j = 0; j < n; ++j) {
        q[j] += _s[i][j] * (alphas[i] - beta);
      }
    }
    var d = q.map((x) => -x).toList();
    if (dot(g, d) > 0) d = g.map((x) => -x).toList();

    final f0 = objective();
    final alpha = lineSearch(w, d, f0, g);
    final s = List.generate(n, (i) => alpha * d[i]);
    final newW = List.generate(n, (i) => w[i] + s[i]);
    weights = newW;

    final gNew = gradient();
    final y = List.generate(n, (i) => gNew[i] - g[i]);
    if (dot(y, s) > 1e-10) {
      _s.add(s);
      _y.add(y);
      if (_s.length > memory) {
        _s.removeAt(0);
        _y.removeAt(0);
      }
    }
    _lastGrad = gNew;
    return objective() <= targetGlobalError;
  }
}

/// Levenberg–Marquardt (damped Gauss–Newton). Excellent for small networks.
class LevenbergMarquardt<P extends VectorSample> extends VectorTrainer<P> {
  double _lambda;

  LevenbergMarquardt(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    double lambda = 0.001,
    double fdEpsilon = 1e-4,
    String? subject,
  }) : _lambda = lambda,
       super(
         ann,
         samplesSet,
         'LevenbergMarquardt',
         fdEpsilon: fdEpsilon,
         subject: subject,
       );

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final w = weights;
    final n = w.length;
    final r = residuals();
    final m = r.length;
    final j = jacobian();

    // JᵀJ (n×n) and Jᵀr (n).
    final jtj = List.generate(n, (_) => List<double>.filled(n, 0));
    final jtr = List<double>.filled(n, 0);
    for (var kk = 0; kk < m; ++kk) {
      final jk = j[kk];
      for (var i = 0; i < n; ++i) {
        jtr[i] += jk[i] * r[kk];
        for (var jj = i; jj < n; ++jj) {
          jtj[i][jj] += jk[i] * jk[jj];
        }
      }
    }
    for (var i = 0; i < n; ++i) {
      for (var jj = 0; jj < i; ++jj) {
        jtj[i][jj] = jtj[jj][i];
      }
    }

    final f0 = objective();
    for (var attempt = 0; attempt < 8; ++attempt) {
      final a = List.generate(n, (i) => List<double>.of(jtj[i]));
      for (var i = 0; i < n; ++i) {
        a[i][i] += _lambda * (jtj[i][i] + 1e-12);
      }
      final b = List.generate(n, (i) => -jtr[i]);
      final delta = solveLinearSystem(a, b);
      if (delta == null) {
        _lambda = min(_lambda * 10, 1e9);
        continue;
      }
      weights = List.generate(n, (i) => w[i] + delta[i]);
      final f = objective();
      if (f < f0) {
        _lambda = max(_lambda * 0.5, 1e-9);
        return f <= targetGlobalError;
      }
      weights = w;
      _lambda = min(_lambda * 10, 1e9);
    }
    weights = w;
    return objective() <= targetGlobalError;
  }
}
