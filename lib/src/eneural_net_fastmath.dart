/*
THIS FAST MATH FUNCTIONS ARE BASED IN THE DART PACKAGE `Complex`:
  - https://pub.dev/packages/complex
  - https://github.com/rwl/complex
  - LICENSE: Apache License - Version 2.0 (http://www.apache.org/licenses/)
 */

import 'dart:math' as dart_math;

import 'eneural_net_fastmath_tables.dart';

/// The `logMaxValue` is the natural logarithm of `doubel.maxFinite`
final logMaxValue = dart_math.log(double.maxFinite);

/// `0x40000000` - used to split a double into two parts, both with the low
/// order bits cleared. Equivalent to `2^30`.
const int hex40000000 = 0x40000000; // 1073741824L

const _f3_4 = 3.0 / 4.0;
const _f1_2 = 1.0 / 2.0;
const _f1_4 = 1.0 / 4.0;

/// This is used by sinQ, because its faster to do a table lookup than
/// a multiply in this time-critical routine
const _eighths = <double>[
  0.0,
  0.125,
  0.25,
  0.375,
  0.5,
  0.625,
  0.75,
  0.875,
  1.0,
  1.125,
  1.25,
  1.375,
  1.5,
  1.625
];

/// Compute the hyperbolic cosine of a number.
double cosh(double x) {
  if (x != x) {
    return x;
  }

  // cosh[z] = (exp(z) + exp(-z))/2

  // for numbers with magnitude 20 or so,
  // exp(-z) can be ignored in comparison with exp(z)

  if (x > 20) {
    if (x >= logMaxValue) {
      // Avoid overflow (MATH-905).
      final t = dart_math.exp(0.5 * x);
      return (0.5 * t) * t;
    } else {
      return 0.5 * expHighPrecision(x);
    }
  } else if (x < -20) {
    if (x <= -logMaxValue) {
      // Avoid overflow (MATH-905).
      final t = expHighPrecision(-0.5 * x);
      return (0.5 * t) * t;
    } else {
      return 0.5 * expHighPrecision(-x);
    }
  }

  //final hiPrec = List<double>(2);
  final hiPrec = List.filled(2, 0.0);
  if (x < 0.0) {
    x = -x;
  }
  expHighPrecision(x, 0.0, hiPrec);

  var ya = hiPrec[0] + hiPrec[1];
  var yb = -(ya - hiPrec[0] - hiPrec[1]);

  var temp = ya * hex40000000;
  final yaa = ya + temp - temp;
  final yab = ya - yaa;

  // recip = 1/y
  final recip = 1.0 / ya;
  temp = recip * hex40000000;
  final recipa = recip + temp - temp;
  var recipb = recip - recipa;

  // Correct for rounding in division
  recipb +=
      (1.0 - yaa * recipa - yaa * recipb - yab * recipa - yab * recipb) * recip;
  // Account for yb
  recipb += -yb * recip * recip;

  // y = y + 1/y
  temp = ya + recipa;
  yb += -(temp - ya - recipa);
  ya = temp;
  temp = ya + recipb;
  yb += -(temp - ya - recipb);
  ya = temp;

  return (ya + yb) * 0.5;
}

/// Internal helper method for exponential function.
///
/// [x] is the original argument of the exponential function.
///
/// NOTE: For higher precision use [expHighPrecision].
double exp(double x) {
  double intPartA;
  double intPartB;
  int intVal;

  // Lookup exp(floor(x)).
  // intPartA will have the upper 22 bits, intPartB will have the lower
  // 52 bits.
  if (x < 0.0) {
    intVal = -x.toInt();

    if (intVal > 746) {
      return 0.0;
    }

    if (intVal > 709) {
      // This will produce a subnormal output
      final result = expHighPrecision(x + 40.19140625) / 285040095144011776.0;
      return result;
    }

    if (intVal == 709) {
      // exp(1.494140625) is nearly a machine number...
      final result = expHighPrecision(x + 1.494140625) / 4.455505956692756620;
      return result;
    }

    intVal++;

    intPartA = expIntTableA[expIntTableMaxIndex - intVal];
    intPartB = expIntTableB[expIntTableMaxIndex - intVal];

    intVal = -intVal;
  } else {
    if (x == double.infinity) {
      return double.infinity;
    }

    intVal = x.toInt();

    if (intVal > 709) {
      return double.infinity;
    }

    intPartA = expIntTableA[expIntTableMaxIndex + intVal];
    intPartB = expIntTableB[expIntTableMaxIndex + intVal];
  }

  // Get the fractional part of x, find the greatest multiple of 2^-10 less than
  // x and look up the exp function of it.
  // fracPartA will have the upper 22 bits, fracPartB the lower 52 bits.
  final intFrac = ((x - intVal) * 1024.0).toInt();
  final fracPartA = expFracTableA[intFrac];
  final fracPartB = expFracTableB[intFrac];

  // epsilon is the difference in x from the nearest multiple of 2^-10.  It
  // has a value in the range 0 <= epsilon < 2^-10.
  // Do the subtraction from x as the last step to avoid possible
  // loss of percison.
  final epsilon = x - (intVal + intFrac / 1024.0);

  // Compute z = exp(epsilon) - 1.0 via a minimax polynomial.  z has
  // full double precision (52 bits).  Since z < 2^-10, we will have
  // 62 bits of precision when combined with the contant 1.  This will be
  // used in the last addition below to get proper rounding.

  // Remez generated polynomial.  Converges on the interval [0, 2^-10], error
  // is less than 0.5 ULP
  var z = 0.04168701738764507;
  z = z * epsilon + 0.1666666505023083;
  z = z * epsilon + 0.5000000000042687;
  z = z * epsilon + 1.0;
  z = z * epsilon + -3.940510424527919E-20;

  // Compute (intPartA+intPartB) * (fracPartA+fracPartB) by binomial
  // expansion.
  // tempA is exact since intPartA and intPartB only have 22 bits each.
  // tempB will have 52 bits of precision.
  final tempA = intPartA * fracPartA;
  final tempB =
      intPartA * fracPartB + intPartB * fracPartA + intPartB * fracPartB;

  // Compute the result.  (1+z)(tempA+tempB).  Order of operations is
  // important.  For accuracy add by increasing size.  tempA is exact and
  // much larger than the others.  If there are extra bits specified from the
  // pow() function, use them.
  final tempC = tempB + tempA;
  var result = tempC * z + tempB + tempA;

  return result;
}

/// Internal helper method for exponential function.
///
/// [x] is the original argument of the exponential function.
/// [extra] bits of precision on input (To Be Confirmed).
/// [highPrecision] extra bits of precision on output (To Be Confirmed)
double expHighPrecision(double x,
    [double extra = 0.0, List<double>? highPrecision]) {
  double intPartA;
  double intPartB;
  int intVal;

  // Lookup exp(floor(x)).
  // intPartA will have the upper 22 bits, intPartB will have the lower
  // 52 bits.
  if (x < 0.0) {
    intVal = -x.toInt();

    if (intVal > 746) {
      if (highPrecision != null) {
        highPrecision[0] = 0.0;
        highPrecision[1] = 0.0;
      }
      return 0.0;
    }

    if (intVal > 709) {
      // This will produce a subnormal output
      final result = expHighPrecision(x + 40.19140625, extra, highPrecision) /
          285040095144011776.0;
      if (highPrecision != null) {
        highPrecision[0] /= 285040095144011776.0;
        highPrecision[1] /= 285040095144011776.0;
      }
      return result;
    }

    if (intVal == 709) {
      // exp(1.494140625) is nearly a machine number...
      final result = expHighPrecision(x + 1.494140625, extra, highPrecision) /
          4.455505956692756620;
      if (highPrecision != null) {
        highPrecision[0] /= 4.455505956692756620;
        highPrecision[1] /= 4.455505956692756620;
      }
      return result;
    }

    intVal++;

    intPartA = expIntTableA[expIntTableMaxIndex - intVal];
    intPartB = expIntTableB[expIntTableMaxIndex - intVal];

    intVal = -intVal;
  } else {
    if (x == double.infinity) {
      if (highPrecision != null) {
        highPrecision[0] = double.infinity;
        highPrecision[1] = 0.0;
      }
      return double.infinity;
    }

    intVal = x.toInt();

    if (intVal > 709) {
      if (highPrecision != null) {
        highPrecision[0] = double.infinity;
        highPrecision[1] = 0.0;
      }
      return double.infinity;
    }

    intPartA = expIntTableA[expIntTableMaxIndex + intVal];
    intPartB = expIntTableB[expIntTableMaxIndex + intVal];
  }

  // Get the fractional part of x, find the greatest multiple of 2^-10 less than
  // x and look up the exp function of it.
  // fracPartA will have the upper 22 bits, fracPartB the lower 52 bits.
  final intFrac = ((x - intVal) * 1024.0).toInt();
  final fracPartA = expFracTableA[intFrac];
  final fracPartB = expFracTableB[intFrac];

  // epsilon is the difference in x from the nearest multiple of 2^-10.  It
  // has a value in the range 0 <= epsilon < 2^-10.
  // Do the subtraction from x as the last step to avoid possible
  // loss of percison.
  final epsilon = x - (intVal + intFrac / 1024.0);

  // Compute z = exp(epsilon) - 1.0 via a minimax polynomial.  z has
  // full double precision (52 bits).  Since z < 2^-10, we will have
  // 62 bits of precision when combined with the contant 1.  This will be
  // used in the last addition below to get proper rounding.

  // Remez generated polynomial.  Converges on the interval [0, 2^-10], error
  // is less than 0.5 ULP
  var z = 0.04168701738764507;
  z = z * epsilon + 0.1666666505023083;
  z = z * epsilon + 0.5000000000042687;
  z = z * epsilon + 1.0;
  z = z * epsilon + -3.940510424527919E-20;

  // Compute (intPartA+intPartB) * (fracPartA+fracPartB) by binomial
  // expansion.
  // tempA is exact since intPartA and intPartB only have 22 bits each.
  // tempB will have 52 bits of precision.
  final tempA = intPartA * fracPartA;
  final tempB =
      intPartA * fracPartB + intPartB * fracPartA + intPartB * fracPartB;

  // Compute the result.  (1+z)(tempA+tempB).  Order of operations is
  // important.  For accuracy add by increasing size.  tempA is exact and
  // much larger than the others.  If there are extra bits specified from the
  // pow() function, use them.
  final tempC = tempB + tempA;
  double result;
  if (extra != 0.0) {
    result = tempC * extra * z + tempC * extra + tempC * z + tempB + tempA;
  } else {
    result = tempC * z + tempB + tempA;
  }

  if (highPrecision != null) {
    // If requesting high precision
    highPrecision[0] = tempA;
    highPrecision[1] = tempC * extra * z + tempC * extra + tempC * z + tempB;
  }

  return result;
}

/// Compute the hyperbolic sine of a number.
double sinh(double x) {
  if (x != x) {
    return x;
  }

  var negate = false;

  // sinh[z] = (exp(z) - exp(-z) / 2

  // for values of z larger than about 20,
  // exp(-z) can be ignored in comparison with exp(z)

  if (x > 20) {
    if (x >= logMaxValue) {
      // Avoid overflow (MATH-905).
      final t = expHighPrecision(0.5 * x);
      return (0.5 * t) * t;
    } else {
      return 0.5 * expHighPrecision(x);
    }
  } else if (x < -20) {
    if (x <= -logMaxValue) {
      // Avoid overflow (MATH-905).
      final t = expHighPrecision(-0.5 * x);
      return (-0.5 * t) * t;
    } else {
      return -0.5 * expHighPrecision(-x);
    }
  }

  if (x == 0) {
    return x;
  }

  if (x < 0.0) {
    x = -x;
    negate = true;
  }

  double result;

  if (x > 0.25) {
    final hiPrec = List.filled(2, 0.0);
    expHighPrecision(x, 0.0, hiPrec);

    var ya = hiPrec[0] + hiPrec[1];
    var yb = -(ya - hiPrec[0] - hiPrec[1]);

    var temp = ya * hex40000000;
    final yaa = ya + temp - temp;
    final yab = ya - yaa;

    // recip = 1/y
    final recip = 1.0 / ya;
    temp = recip * hex40000000;
    var recipa = recip + temp - temp;
    var recipb = recip - recipa;

    // Correct for rounding in division
    recipb +=
        (1.0 - yaa * recipa - yaa * recipb - yab * recipa - yab * recipb) *
            recip;
    // Account for yb
    recipb += -yb * recip * recip;

    recipa = -recipa;
    recipb = -recipb;

    // y = y + 1/y
    temp = ya + recipa;
    yb += -(temp - ya - recipa);
    ya = temp;
    temp = ya + recipb;
    yb += -(temp - ya - recipb);
    ya = temp;

    result = ya + yb;
    result *= 0.5;
  } else {
    final hiPrec = List.filled(2, 0.0);
    expm1(x, hiPrec);

    var ya = hiPrec[0] + hiPrec[1];
    var yb = -(ya - hiPrec[0] - hiPrec[1]);

    /* Compute expm1(-x) = -expm1(x) / (expm1(x) + 1) */
    final denom = 1.0 + ya;
    final denomr = 1.0 / denom;
    final denomb = -(denom - 1.0 - ya) + yb;
    final ratio = ya * denomr;
    var temp = ratio * hex40000000;
    final ra = ratio + temp - temp;
    var rb = ratio - ra;

    temp = denom * hex40000000;
    final za = denom + temp - temp;
    final zb = denom - za;

    rb += (ya - za * ra - za * rb - zb * ra - zb * rb) * denomr;

    // Adjust for yb
    rb += yb * denomr; // numerator
    rb += -ya * denomb * denomr * denomr; // denominator

    // y = y - 1/y
    temp = ya + ra;
    yb += -(temp - ya - ra);
    ya = temp;
    temp = ya + rb;
    yb += -(temp - ya - rb);
    ya = temp;

    result = ya + yb;
    result *= 0.5;
  }

  if (negate) {
    result = -result;
  }

  return result;
}

/// Internal helper function to compute arctangent.
///
/// [xa] number from which arctangent is requested.
/// [xb] extra bits for x (may be 0.0).
/// [leftPlane] if true, result angle must be put in the left half plane.
/// Returns `atan(xa + xb)` (or angle shifted by `PI` if leftPlane is true)
double atan(double xa, [double xb = 0.0, bool leftPlane = false]) {
  if (xa == 0.0) {
    // Matches +/- 0.0; return correct sign
    return leftPlane ? copySign(dart_math.pi, xa) : xa;
  }

  bool negate;
  if (xa < 0) {
    // negative
    xa = -xa;
    xb = -xb;
    negate = true;
  } else {
    negate = false;
  }

  if (xa > 1.633123935319537E16) {
    // Very large input
    return (negate != leftPlane)
        ? (-dart_math.pi * _f1_2)
        : (dart_math.pi * _f1_2);
  }

  // Estimate the closest tabulated arctan value, compute eps = xa-tangentTable
  int idx;
  if (xa < 1) {
    idx = (((-1.7168146928204136 * xa * xa + 8.0) * xa) + 0.5).toInt();
  } else {
    final oneOverXa = 1 / xa;
    idx = (-((-1.7168146928204136 * oneOverXa * oneOverXa + 8.0) * oneOverXa) +
            13.07)
        .toInt();
  }

  final ttA = tangentTableA[idx];
  final ttB = tangentTableB[idx];

  var epsA = xa - ttA;
  var epsB = -(epsA - xa + ttA);
  epsB += xb - ttB;

  var temp = epsA + epsB;
  epsB = -(temp - epsA - epsB);
  epsA = temp;

  /* Compute eps = eps / (1.0 + xa*tangent) */
  temp = xa * hex40000000;
  var ya = xa + temp - temp;
  var yb = xb + xa - ya;
  xa = ya;
  xb += yb;

  //if (idx > 8 || idx == 0)
  if (idx == 0) {
    /// If the slope of the arctan is gentle enough (< 0.45),
    /// this approximation will suffice
    //double denom = 1.0 / (1.0 + xa*tangentTableA[idx] + xb*tangentTableA[idx] + xa*tangentTableB[idx] + xb*tangentTableB[idx]);
    final denom = 1.0 / (1.0 + (xa + xb) * (ttA + ttB));
    //double denom = 1.0 / (1.0 + xa*tangentTableA[idx]);
    ya = epsA * denom;
    yb = epsB * denom;
  } else {
    var temp2 = xa * ttA;
    var za = 1.0 + temp2;
    var zb = -(za - 1.0 - temp2);
    temp2 = xb * ttA + xa * ttB;
    temp = za + temp2;
    zb += -(temp - za - temp2);
    za = temp;

    zb += xb * ttB;
    ya = epsA / za;

    temp = ya * hex40000000;
    final yaa = (ya + temp) - temp;
    final yab = ya - yaa;

    temp = za * hex40000000;
    final zaa = (za + temp) - temp;
    final zab = za - zaa;

    /* Correct for rounding in division */
    yb = (epsA - yaa * zaa - yaa * zab - yab * zaa - yab * zab) / za;

    yb += -epsA * zb / za / za;
    yb += epsB / za;
  }

  epsA = ya;
  epsB = yb;

  // Evaluate polynomial
  final epsA2 = epsA * epsA;

  /*
  yb = -0.09001346640161823;
  yb = yb * epsA2 + 0.11110718400605211;
  yb = yb * epsA2 + -0.1428571349122913;
  yb = yb * epsA2 + 0.19999999999273194;
  yb = yb * epsA2 + -0.33333333333333093;
  yb = yb * epsA2 * epsA;
  */

  yb = 0.07490822288864472;
  yb = yb * epsA2 - 0.09088450866185192;
  yb = yb * epsA2 + 0.11111095942313305;
  yb = yb * epsA2 - 0.1428571423679182;
  yb = yb * epsA2 + 0.19999999999923582;
  yb = yb * epsA2 - 0.33333333333333287;
  yb = yb * epsA2 * epsA;

  ya = epsA;

  temp = ya + yb;
  yb = -(temp - ya - yb);
  ya = temp;

  /* Add in effect of epsB.   atan'(x) = 1/(1+x^2) */
  yb += epsB / (1.0 + epsA * epsA);

  final eighths = _eighths[idx];

  //result = yb + eighths[idx] + ya;
  var za = eighths + ya;
  var zb = -(za - eighths - ya);
  temp = za + yb;
  zb += -(temp - za - yb);
  za = temp;

  var result = za + zb;

  if (leftPlane) {
    // Result is in the left plane
    final resultb = -(result - za - zb);
    final pia = 1.5707963267948966 * 2;
    final pib = 6.123233995736766E-17 * 2;

    za = pia - result;
    zb = -(za - pia + result);
    zb += pib - resultb;

    result = za + zb;
  }

  if (negate != leftPlane) {
    result = -result;
  }

  return result;
}

/// Compute `exp(x) - 1`.
double expm1(double x, List<double> hiPrecOut) {
  if (x != x || x == 0.0) {
    // NaN or zero
    return x;
  }

  if (x <= -1.0 || x >= 1.0) {
    // If not between +/- 1.0
    //return exp(x) - 1.0;
    final hiPrec = List.filled(2, 0.0);
    expHighPrecision(x, 0.0, hiPrec);
    if (x > 0.0) {
      return -1.0 + hiPrec[0] + hiPrec[1];
    } else {
      final ra = -1.0 + hiPrec[0];
      var rb = -(ra + 1.0 - hiPrec[0]);
      rb += hiPrec[1];
      return ra + rb;
    }
  }

  double baseA;
  double baseB;
  double epsilon;
  var negative = false;

  if (x < 0.0) {
    x = -x;
    negative = true;
  }

  {
    final intFrac = (x * 1024.0).toInt();
    var tempA = expFracTableA[intFrac] - 1.0;
    var tempB = expFracTableB[intFrac];

    var temp = tempA + tempB;
    tempB = -(temp - tempA - tempB);
    tempA = temp;

    temp = tempA * hex40000000;
    baseA = tempA + temp - temp;
    baseB = tempB + (tempA - baseA);

    epsilon = x - intFrac / 1024.0;
  }

  /// Compute expm1(epsilon)
  var zb = 0.008336750013465571;
  zb = zb * epsilon + 0.041666663879186654;
  zb = zb * epsilon + 0.16666666666745392;
  zb = zb * epsilon + 0.49999999999999994;
  zb *= epsilon;
  zb *= epsilon;

  var za = epsilon;
  var temp = za + zb;
  zb = -(temp - za - zb);
  za = temp;

  temp = za * hex40000000;
  temp = za + temp - temp;
  zb += za - temp;
  za = temp;

  /// Combine the parts.
  /// `expm1(a+b) = expm1(a) + expm1(b) + expm1(a)*expm1(b)`
  var ya = za * baseA;
  //double yb = za*baseB + zb*baseA + zb*baseB;
  temp = ya + za * baseB;
  var yb = -(temp - ya - za * baseB);
  ya = temp;

  temp = ya + zb * baseA;
  yb += -(temp - ya - zb * baseA);
  ya = temp;

  temp = ya + zb * baseB;
  yb += -(temp - ya - zb * baseB);
  ya = temp;

  //ya = ya + za + baseA;
  //yb = yb + zb + baseB;
  temp = ya + baseA;
  yb += -(temp - baseA - ya);
  ya = temp;

  temp = ya + za;
  //yb += (ya > za) ? -(temp - ya - za) : -(temp - za - ya);
  yb += -(temp - ya - za);
  ya = temp;

  temp = ya + baseB;
  //yb += (ya > baseB) ? -(temp - ya - baseB) : -(temp - baseB - ya);
  yb += -(temp - ya - baseB);
  ya = temp;

  temp = ya + zb;
  //yb += (ya > zb) ? -(temp - ya - zb) : -(temp - zb - ya);
  yb += -(temp - ya - zb);
  ya = temp;

  if (negative) {
    /// Compute `expm1(-x) = -expm1(x) / (expm1(x) + 1)`
    final denom = 1.0 + ya;
    final denomr = 1.0 / denom;
    final denomb = -(denom - 1.0 - ya) + yb;
    final ratio = ya * denomr;
    temp = ratio * hex40000000;
    final ra = ratio + temp - temp;
    var rb = ratio - ra;

    temp = denom * hex40000000;
    za = denom + temp - temp;
    zb = denom - za;

    rb += (ya - za * ra - za * rb - zb * ra - zb * rb) * denomr;

    // f(x) = x/1+x
    // Compute f'(x)
    // Product rule:  d(uv) = du*v + u*dv
    // Chain rule:  d(f(g(x)) = f'(g(x))*f(g'(x))
    // d(1/x) = -1/(x*x)
    // d(1/1+x) = -1/( (1+x)^2) *  1 =  -1/((1+x)*(1+x))
    // d(x/1+x) = -x/((1+x)(1+x)) + 1/1+x = 1 / ((1+x)(1+x))

    // Adjust for yb
    rb += yb * denomr; // numerator
    rb += -ya * denomb * denomr * denomr; // denominator

    // negate
    ya = -ra;
    yb = -rb;
  }

  // TODO(@kranfix): remove?
  //if (hiPrecOut != null) {
  //  hiPrecOut[0] = ya;
  //  hiPrecOut[1] = yb;
  //}

  return ya + yb;
}

/// Two arguments arctangent function
///
/// [y] ordinate. [x] abscissa.
/// Returns phase angle of point (x,y) between `-PI` and `PI`.
double atan2(double y, double x) {
  if (x != x || y != y) {
    return double.nan;
  }

  if (y == 0) {
    final result = x * y;
    final invx = 1.0 / x;
    final invy = 1.0 / y;

    if (invx == 0) {
      // X is infinite
      if (x > 0) {
        return y; // return +/- 0.0
      } else {
        return copySign(dart_math.pi, y);
      }
    }

    if (x < 0 || invx < 0) {
      if (y < 0 || invy < 0) {
        return -dart_math.pi;
      } else {
        return dart_math.pi;
      }
    } else {
      return result;
    }
  }

  // y cannot now be zero

  if (y == double.infinity) {
    if (x == double.infinity) {
      return dart_math.pi * _f1_4;
    }

    if (x == double.negativeInfinity) {
      return dart_math.pi * _f3_4;
    }

    return dart_math.pi * _f1_2;
  }

  if (y == double.negativeInfinity) {
    if (x == double.infinity) {
      return -dart_math.pi * _f1_4;
    }

    if (x == double.negativeInfinity) {
      return -dart_math.pi * _f3_4;
    }

    return -dart_math.pi * _f1_2;
  }

  if (x == double.infinity) {
    if (y > 0 || 1 / y > 0) {
      return 0.0;
    }

    if (y < 0 || 1 / y < 0) {
      return -0.0;
    }
  }

  if (x == double.negativeInfinity) {
    if (y > 0.0 || 1 / y > 0.0) {
      return dart_math.pi;
    }

    if (y < 0 || 1 / y < 0) {
      return -dart_math.pi;
    }
  }

  // Neither y nor x can be infinite or NAN here

  if (x == 0) {
    if (y > 0 || 1 / y > 0) {
      return dart_math.pi * _f1_2;
    }

    if (y < 0 || 1 / y < 0) {
      return -dart_math.pi * _f1_2;
    }
  }

  // Compute ratio r = y/x
  final r = y / x;
  if (r.isInfinite) {
    // bypass calculations that can create NaN
    return atan(r, 0.0, x < 0);
  }

  var ra = r; // TODO(rwl): doubleHighPart(r);
  var rb = r - ra;

  // Split x
  final xa = x; // TODO(rwl): doubleHighPart(x);
  final xb = x - xa;

  rb += (y - ra * xa - ra * xb - rb * xa - rb * xb) / x;

  final temp = ra + rb;
  rb = -(temp - ra - rb);
  ra = temp;

  if (ra == 0) {
    // Fix up the sign so atan works correctly
    ra = copySign(0.0, y);
  }

  // Call atan
  return atan(ra, rb, x < 0);
}

/// Returns the first argument with the sign of the second argument.
/// A NaN `sign` argument is treated as positive.
///
/// [magnitude] the value to return.
/// [sign] the sign for the returned value.
/// Returns the magnitude with the same sign as the `sign` argument.
double copySign(double magnitude, double sign) {
  // The highest order bit is going to be zero if the
  // highest order bit of m and s is the same and one otherwise.
  // So (m^s) will be positive if both m and s have the same sign
  // and negative otherwise.
  /*final long m = Double.doubleToRawLongBits(magnitude); // don't care about NaN
  final long s = Double.doubleToRawLongBits(sign);
  if ((m^s) >= 0) {
      return magnitude;
  }
  return -magnitude; // flip sign*/
  if (sign == 0.0 || sign.isNaN || magnitude.sign == sign.sign) {
    return magnitude;
  }
  return -magnitude; // flip sign
}
