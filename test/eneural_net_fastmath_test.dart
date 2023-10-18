import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net_fast_math.dart' as fast_math;
import 'package:eneural_net/eneural_net_extensions.dart';

import 'package:test/test.dart';

void main() {
  group('Fast Math', () {
    setUp(() {});

    test('exp', () {
      for (var v = -10.0; v <= 10; v += 0.10) {
        var e1 = exp(v);
        var e2 = fast_math.exp(v);
        var error = e1 - e2;
        expect(error.abs() < 4.0e-14, isTrue,
            reason: '$v -> $e1 - $e2 = $error');
      }
    });

    test('expFloat32x4 -10..10', () {
      var limit = 10.0;

      for (var v = -limit; v <= limit; v += 0.10) {
        var v1 = v;
        var v2 = v + 0.01;
        var v3 = (-v) - 0.02;
        var v4 = (-v) - 0.03;

        var e1 = fast_math.exp(v1);
        var e2 = fast_math.exp(v2);
        var e3 = fast_math.exp(v3);
        var e4 = fast_math.exp(v4);

        var e = Float32x4(e1, e2, e3, e4);

        var vx4 = Float32x4(v1, v2, v3, v4);

        var ex4 = fast_math.expFloat32x4(vx4);

        var error = e - ex4;
        expect(error.abs().maxInLane < 0.01, isTrue,
            reason: '$vx4 -> $e - $ex4 = $error');
      }
    });

    test('expFloat32x4 -16777546..16777546', () {
      var limit = 16777546.0;

      for (var v = -limit; v <= limit; v += 10.10) {
        var v1 = v;
        var v2 = v + 0.01;
        var v3 = (-v) - 0.02;
        var v4 = (-v) - 0.03;

        var vx4 = Float32x4(v1, v2, v3, v4);
        fast_math.expFloat32x4(vx4);
      }
    });
  });
}
