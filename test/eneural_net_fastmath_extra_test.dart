import 'dart:math' as dart_math;
import 'dart:typed_data';

import 'package:eneural_net/eneural_net_fast_math.dart' as fast_math;
import 'package:test/test.dart';

/// Relative comparison, so the tolerance scales with the magnitude.
void expectClose(double actual, double expected, {double relative = 1e-12}) {
  var tolerance = (expected.abs() * relative).clamp(1e-15, double.infinity);
  expect(
    actual,
    closeTo(expected, tolerance),
    reason: 'expected $expected, got $actual',
  );
}

void main() {
  group('fast_math.exp', () {
    test('reference values', () {
      expectClose(fast_math.exp(0), 1.0);
      expectClose(fast_math.exp(1), dart_math.e);
      expectClose(fast_math.exp(-1), 1 / dart_math.e);
    });

    test('matches dart:math over a wide range', () {
      for (var v = -50.0; v <= 50.0; v += 0.37) {
        expectClose(fast_math.exp(v), dart_math.exp(v), relative: 1e-11);
      }
    });

    test('handles the special values', () {
      expect(fast_math.exp(double.nan).isNaN, isTrue);
      expect(fast_math.exp(double.infinity), equals(double.infinity));
      expect(fast_math.exp(double.negativeInfinity), equals(0.0));
    });

    test('clamps finite arguments to -87..87', () {
      // Documented behavior: the argument is bounded so that the result fits
      // a `Float32x4` lane. Use `expHighPrecision` for the full range.
      expect(fast_math.exp(1000), equals(fast_math.exp(87)));
      expect(fast_math.exp(-1000), equals(fast_math.exp(-87)));

      expect(fast_math.exp(87).isFinite, isTrue);
      expect(fast_math.exp(-87) > 0, isTrue);

      // The unbounded variant does overflow/underflow:
      expect(fast_math.expHighPrecision(1000).isInfinite, isTrue);
      expect(fast_math.expHighPrecision(-1000), equals(0.0));
    });

    test('near the overflow/underflow thresholds', () {
      for (var v in [-750.0, -746.0, -709.0, 709.0, 710.0, 750.0]) {
        var r = fast_math.exp(v);
        expect(r.isNaN, isFalse, reason: 'exp($v) is NaN');
        expect(r >= 0, isTrue, reason: 'exp($v) is negative');
      }
    });

    test('is monotonically increasing', () {
      var previous = 0.0;
      for (var v = -20.0; v <= 20.0; v += 0.5) {
        var r = fast_math.exp(v);
        expect(r >= previous, isTrue, reason: 'not increasing at $v');
        previous = r;
      }
    });
  });

  group('fast_math.expHighPrecision', () {
    test('matches exp when no extra precision is given', () {
      for (var v = -10.0; v <= 10.0; v += 0.5) {
        expectClose(
          fast_math.expHighPrecision(v),
          dart_math.exp(v),
          relative: 1e-11,
        );
      }
    });

    test('fills the high precision output', () {
      var out = <double>[0.0, 0.0];
      var r = fast_math.expHighPrecision(1.0, 0.0, out);

      expectClose(r, dart_math.e, relative: 1e-11);
      expectClose(out[0] + out[1], dart_math.e, relative: 1e-11);
    });

    test('handles extreme values', () {
      expect(fast_math.expHighPrecision(double.nan).isNaN, isTrue);
      expect(
        fast_math.expHighPrecision(double.infinity),
        equals(double.infinity),
      );
      expect(fast_math.expHighPrecision(double.negativeInfinity), equals(0.0));
      expect(fast_math.expHighPrecision(1000).isInfinite, isTrue);
      expect(fast_math.expHighPrecision(-1000), equals(0.0));
    });

    test('accepts an extra precision term', () {
      var r = fast_math.expHighPrecision(1.0, 1.0e-16);
      expectClose(r, dart_math.e, relative: 1e-11);
    });
  });

  group('fast_math.expFloat32x4', () {
    test('matches exp on each lane', () {
      for (var v = -10.0; v <= 10.0; v += 0.5) {
        var entry = Float32x4(v, v + 0.1, -v, -v - 0.1);
        var result = fast_math.expFloat32x4(entry);

        expect(result.x, closeTo(dart_math.exp(v), dart_math.exp(v) * 1e-3));
        expect(result.z, closeTo(dart_math.exp(-v), dart_math.exp(-v) * 1e-3));
      }
    });

    test('saturates outside the representable range', () {
      var high = fast_math.expFloat32x4(Float32x4.splat(1000));
      expect(high.x.isNaN, isFalse);

      var low = fast_math.expFloat32x4(Float32x4.splat(-1000));
      expect(low.x, closeTo(0.0, 1e-6));
    });

    test('exp(0) is 1 on every lane', () {
      var r = fast_math.expFloat32x4(Float32x4.zero());
      for (var lane in [r.x, r.y, r.z, r.w]) {
        expect(lane, closeTo(1.0, 1e-5));
      }
    });
  });

  group('fast_math.expm1', () {
    test('matches exp(x)-1', () {
      var out = <double>[0.0, 0.0];

      for (var v = -5.0; v <= 5.0; v += 0.25) {
        var r = fast_math.expm1(v, out);
        expectClose(r, dart_math.exp(v) - 1, relative: 1e-9);
      }
    });

    test('is accurate near zero', () {
      var out = <double>[0.0, 0.0];

      for (var v in [1e-3, 1e-6, 1e-9, -1e-3, -1e-6, -1e-9]) {
        var r = fast_math.expm1(v, out);

        // `exp(v) - 1` loses precision near zero, so `expm1(v) ~= v`:
        expectClose(r, v, relative: 1e-2);
        expect(r.sign, equals(v.sign), reason: 'sign of expm1($v)');
      }
    });

    test('fills the high precision output', () {
      var out = <double>[0.0, 0.0];

      for (var v in [0.1, 0.5, -0.1, -0.5]) {
        out[0] = 0.0;
        out[1] = 0.0;

        var r = fast_math.expm1(v, out);

        expectClose(out[0] + out[1], r, relative: 1e-9);
        expect(out[0], isNot(equals(0.0)), reason: 'hiPrecOut of expm1($v)');
      }
    });

    test('reference values', () {
      var out = <double>[0.0, 0.0];

      expect(fast_math.expm1(0.0, out), equals(0.0));
      expectClose(fast_math.expm1(1.0, out), dart_math.e - 1, relative: 1e-9);
    });

    test('handles extreme values', () {
      var out = <double>[0.0, 0.0];

      expect(fast_math.expm1(double.nan, out).isNaN, isTrue);
      expect(fast_math.expm1(double.infinity, out), equals(double.infinity));
      expect(fast_math.expm1(-double.infinity, out), equals(-1.0));
      expect(fast_math.expm1(1000, out).isInfinite, isTrue);
      expect(fast_math.expm1(-1000, out), equals(-1.0));
    });
  });

  group('fast_math.cosh', () {
    test('reference values', () {
      expectClose(fast_math.cosh(0), 1.0);
      expectClose(fast_math.cosh(1), (dart_math.e + 1 / dart_math.e) / 2);
    });

    test('is even: cosh(-x) == cosh(x)', () {
      for (var v = 0.0; v <= 20.0; v += 0.5) {
        expectClose(fast_math.cosh(-v), fast_math.cosh(v), relative: 1e-12);
      }
    });

    test('matches the exponential definition', () {
      for (var v = -20.0; v <= 20.0; v += 0.7) {
        var expected = (dart_math.exp(v) + dart_math.exp(-v)) / 2;
        expectClose(fast_math.cosh(v), expected, relative: 1e-9);
      }
    });

    test('is never below 1', () {
      for (var v = -30.0; v <= 30.0; v += 0.3) {
        expect(fast_math.cosh(v) >= 1.0, isTrue, reason: 'cosh($v)');
      }
    });

    test('handles extreme values', () {
      expect(fast_math.cosh(double.nan).isNaN, isTrue);
      expect(fast_math.cosh(double.infinity), equals(double.infinity));
      expect(fast_math.cosh(double.negativeInfinity), equals(double.infinity));
      expect(fast_math.cosh(1000).isInfinite, isTrue);
      expect(fast_math.cosh(-1000).isInfinite, isTrue);
    });

    test('large but finite values', () {
      for (var v in [-710.0, -700.0, 700.0, 710.0]) {
        var r = fast_math.cosh(v);
        expect(r.isNaN, isFalse, reason: 'cosh($v)');
        expect(r > 0, isTrue, reason: 'cosh($v)');
      }
    });
  });

  group('fast_math.sinh', () {
    test('reference values', () {
      expect(fast_math.sinh(0), equals(0.0));
      expectClose(fast_math.sinh(1), (dart_math.e - 1 / dart_math.e) / 2);
    });

    test('is odd: sinh(-x) == -sinh(x)', () {
      for (var v = 0.0; v <= 20.0; v += 0.5) {
        expectClose(fast_math.sinh(-v), -fast_math.sinh(v), relative: 1e-12);
      }
    });

    test('matches the exponential definition', () {
      for (var v = -20.0; v <= 20.0; v += 0.7) {
        var expected = (dart_math.exp(v) - dart_math.exp(-v)) / 2;
        expectClose(fast_math.sinh(v), expected, relative: 1e-9);
      }
    });

    test('is not zero for small arguments', () {
      // `sinh` uses the high precision output of `expm1` for |x| <= 0.25.
      for (var v in [1e-9, 1e-6, 1e-3, 0.1, 0.2, 0.25, -1e-3, -0.1, -0.25]) {
        var r = fast_math.sinh(v);
        var expected = (dart_math.exp(v) - dart_math.exp(-v)) / 2;

        expect(r, isNot(equals(0.0)), reason: 'sinh($v) collapsed to zero');
        expect(r.sign, equals(v.sign), reason: 'sign of sinh($v)');
        expectClose(r, expected, relative: 1e-4);
      }
    });

    test('matches the reference over the small-argument branch', () {
      for (var v = -0.25; v <= 0.25; v += 0.01) {
        var expected = (dart_math.exp(v) - dart_math.exp(-v)) / 2;
        expectClose(fast_math.sinh(v), expected, relative: 1e-4);
      }
    });

    test('handles extreme values', () {
      expect(fast_math.sinh(double.nan).isNaN, isTrue);
      expect(fast_math.sinh(double.infinity), equals(double.infinity));
      expect(
        fast_math.sinh(double.negativeInfinity),
        equals(double.negativeInfinity),
      );
      expect(fast_math.sinh(1000).isInfinite, isTrue);
      expect(fast_math.sinh(-1000).isInfinite, isTrue);
    });

    test('large but finite values', () {
      for (var v in [-710.0, -700.0, -0.5, 0.5, 700.0, 710.0]) {
        var r = fast_math.sinh(v);
        expect(r.isNaN, isFalse, reason: 'sinh($v)');
      }
    });

    test('cosh² - sinh² == 1', () {
      for (var v = -10.0; v <= 10.0; v += 0.5) {
        var c = fast_math.cosh(v);
        var s = fast_math.sinh(v);
        expectClose((c * c) - (s * s), 1.0, relative: 1e-6);
      }
    });
  });

  group('fast_math.atan', () {
    test('reference values', () {
      expect(fast_math.atan(0), equals(0.0));
      expectClose(fast_math.atan(1), dart_math.pi / 4, relative: 1e-12);
      expectClose(fast_math.atan(-1), -dart_math.pi / 4, relative: 1e-12);
    });

    test('matches dart:math', () {
      for (var v = -20.0; v <= 20.0; v += 0.37) {
        expectClose(fast_math.atan(v), dart_math.atan(v), relative: 1e-10);
      }
    });

    test('is odd: atan(-x) == -atan(x)', () {
      for (var v = 0.0; v <= 10.0; v += 0.5) {
        expectClose(fast_math.atan(-v), -fast_math.atan(v), relative: 1e-12);
      }
    });

    test('is bounded by +-pi/2', () {
      for (var v in [-1e9, -1e3, -1.0, 1.0, 1e3, 1e9]) {
        expect(fast_math.atan(v).abs() <= dart_math.pi / 2, isTrue);
      }
    });

    test('handles extreme values', () {
      expect(fast_math.atan(double.nan).isNaN, isTrue);
      expectClose(fast_math.atan(double.infinity), dart_math.pi / 2);
      expectClose(fast_math.atan(double.negativeInfinity), -dart_math.pi / 2);
    });

    test('is accurate for tiny values', () {
      for (var v in [1e-8, 1e-10, -1e-8]) {
        expectClose(fast_math.atan(v), v, relative: 1e-6);
      }
    });

    test('the leftPlane variant shifts by pi', () {
      // Used by `atan2` to compute the angle on the left half-plane.
      var right = fast_math.atan(1.0, 0.0, false);
      var left = fast_math.atan(1.0, 0.0, true);

      expectClose(left, right - dart_math.pi, relative: 1e-12);
    });

    test('accepts an extra precision term', () {
      expectClose(
        fast_math.atan(1.0, 1e-17),
        dart_math.atan(1.0),
        relative: 1e-10,
      );
    });
  });

  group('fast_math.atan2', () {
    test('reference values', () {
      expect(fast_math.atan2(0, 1), equals(0.0));
      expectClose(fast_math.atan2(1, 0), dart_math.pi / 2);
      expectClose(fast_math.atan2(-1, 0), -dart_math.pi / 2);
      expectClose(fast_math.atan2(0, -1), dart_math.pi);
    });

    test('matches dart:math across the quadrants', () {
      for (var y = -5.0; y <= 5.0; y += 0.7) {
        for (var x = -5.0; x <= 5.0; x += 0.7) {
          if (x == 0 && y == 0) continue;
          expectClose(
            fast_math.atan2(y, x),
            dart_math.atan2(y, x),
            relative: 1e-9,
          );
        }
      }
    });

    test('handles the zero cases', () {
      expect(fast_math.atan2(0.0, 0.0), equals(0.0));
      expectClose(fast_math.atan2(0.0, -1.0), dart_math.pi);
      expect(fast_math.atan2(0.0, 1.0), equals(0.0));

      // Negative zero:
      expect(fast_math.atan2(-0.0, 1.0), equals(dart_math.atan2(-0.0, 1.0)));
      expectClose(fast_math.atan2(-0.0, -1.0), -dart_math.pi);
    });

    test('handles infinities', () {
      expect(fast_math.atan2(double.nan, 1).isNaN, isTrue);
      expect(fast_math.atan2(1, double.nan).isNaN, isTrue);

      expectClose(fast_math.atan2(double.infinity, 1), dart_math.pi / 2);
      expectClose(
        fast_math.atan2(double.negativeInfinity, 1),
        -dart_math.pi / 2,
      );

      expectClose(
        fast_math.atan2(double.infinity, double.infinity),
        dart_math.pi / 4,
      );
      expectClose(
        fast_math.atan2(double.infinity, double.negativeInfinity),
        dart_math.pi * 3 / 4,
      );
      expectClose(
        fast_math.atan2(double.negativeInfinity, double.infinity),
        -dart_math.pi / 4,
      );
      expectClose(
        fast_math.atan2(double.negativeInfinity, double.negativeInfinity),
        -dart_math.pi * 3 / 4,
      );

      expect(fast_math.atan2(1.0, double.infinity), equals(0.0));
      expectClose(fast_math.atan2(1.0, double.negativeInfinity), dart_math.pi);
      expectClose(
        fast_math.atan2(-1.0, double.negativeInfinity),
        -dart_math.pi,
      );
    });

    test('handles very small and very large ratios', () {
      expectClose(
        fast_math.atan2(1e-300, 1e300),
        dart_math.atan2(1e-300, 1e300),
        relative: 1e-6,
      );
      expectClose(
        fast_math.atan2(1e300, 1e-300),
        dart_math.atan2(1e300, 1e-300),
        relative: 1e-6,
      );
    });
  });

  group('fast_math.copySign', () {
    test('copies the sign of the second argument', () {
      expect(fast_math.copySign(3.0, 1.0), equals(3.0));
      expect(fast_math.copySign(3.0, -1.0), equals(-3.0));
      expect(fast_math.copySign(-3.0, 1.0), equals(3.0));
      expect(fast_math.copySign(-3.0, -1.0), equals(-3.0));
    });

    test('handles zeroes and infinities', () {
      expect(fast_math.copySign(0.0, -1.0), equals(0.0));
      expect(fast_math.copySign(1.0, 0.0), equals(1.0));
      expect(
        fast_math.copySign(double.infinity, -1.0),
        equals(double.negativeInfinity),
      );
    });

    test('a NaN sign keeps the magnitude', () {
      expect(fast_math.copySign(3.0, double.nan).abs(), equals(3.0));
    });
  });

  group('fast_math: constants', () {
    test('logMaxValue', () {
      expect(fast_math.logMaxValue, equals(dart_math.log(double.maxFinite)));
    });

    test('hex40000000', () {
      expect(fast_math.hex40000000, equals(0x40000000));
      expect(fast_math.hex40000000, equals(1073741824));
    });
  });
}
