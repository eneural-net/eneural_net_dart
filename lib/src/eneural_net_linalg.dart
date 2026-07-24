/// Small dense linear-algebra helpers for the second-order trainers.
library;

/// Solves `A·x = b` (A is `n×n`, row-major as `List<List<double>>`) by Gaussian
/// elimination with partial pivoting. Returns `x`, or `null` if `A` is singular.
List<double>? solveLinearSystem(List<List<double>> a, List<double> b) {
  final n = b.length;
  // Work on copies.
  final m = List.generate(n, (i) => List<double>.of(a[i]));
  final rhs = List<double>.of(b);

  for (var col = 0; col < n; ++col) {
    // Partial pivot.
    var pivot = col;
    var maxAbs = m[col][col].abs();
    for (var r = col + 1; r < n; ++r) {
      final v = m[r][col].abs();
      if (v > maxAbs) {
        maxAbs = v;
        pivot = r;
      }
    }
    if (maxAbs < 1e-30) return null; // singular
    if (pivot != col) {
      final tmp = m[col];
      m[col] = m[pivot];
      m[pivot] = tmp;
      final tb = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = tb;
    }

    final diag = m[col][col];
    for (var r = col + 1; r < n; ++r) {
      final factor = m[r][col] / diag;
      if (factor == 0) continue;
      for (var c = col; c < n; ++c) {
        m[r][c] -= factor * m[col][c];
      }
      rhs[r] -= factor * rhs[col];
    }
  }

  // Back-substitution.
  final x = List<double>.filled(n, 0);
  for (var row = n - 1; row >= 0; --row) {
    var sum = rhs[row];
    for (var c = row + 1; c < n; ++c) {
      sum -= m[row][c] * x[c];
    }
    x[row] = sum / m[row][row];
  }
  return x;
}

/// Dot product of two vectors.
double dot(List<double> a, List<double> b) {
  var s = 0.0;
  for (var i = 0; i < a.length; ++i) {
    s += a[i] * b[i];
  }
  return s;
}
