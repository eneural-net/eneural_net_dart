import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

/// The `Float32x4` activation functions of the library.
List<ActivationFunctionFloat32x4> allFloat32x4Functions() => [
  ActivationFunctionLinear(),
  ActivationFunctionSigmoid(),
  ActivationFunctionSigmoidFast(),
  ActivationFunctionSigmoidBoundedFast(),
  ActivationFunctionSigmoidBoundedFast(scale: 3),
];

void main() {
  group('ActivationFunction: SIMD matches the scalar version', () {
    test('activateEntry == activate for every lane', () {
      for (var af in allFloat32x4Functions()) {
        for (var x = -10.0; x <= 10.0; x += 0.25) {
          var scalar = af.activate(x);
          var simd = af.activateEntry(Float32x4.splat(x));

          for (var lane in [simd.x, simd.y, simd.z, simd.w]) {
            expect(
              lane,
              closeTo(scalar, 1e-6),
              reason: '${af.name}.activateEntry($x) != activate($x)',
            );
          }
        }
      }
    });

    test('activateEntry computes each lane independently', () {
      for (var af in allFloat32x4Functions()) {
        var entry = af.activateEntry(Float32x4(-2, -0.5, 0.5, 2));

        expect(entry.x, closeTo(af.activate(-2), 1e-6), reason: af.name);
        expect(entry.y, closeTo(af.activate(-0.5), 1e-6), reason: af.name);
        expect(entry.z, closeTo(af.activate(0.5), 1e-6), reason: af.name);
        expect(entry.w, closeTo(af.activate(2), 1e-6), reason: af.name);
      }
    });

    test('derivativeEntry == derivative for every lane', () {
      for (var af in allFloat32x4Functions()) {
        for (var o = 0.0; o <= 1.0; o += 0.05) {
          var scalar = af.derivative(o);
          var simd = af.derivativeEntry(Float32x4.splat(o));

          expect(simd.x, closeTo(scalar, 1e-6), reason: '${af.name} at $o');
          expect(simd.w, closeTo(scalar, 1e-6), reason: '${af.name} at $o');
        }
      }
    });

    test('derivativeEntryWithFlatSpot == derivativeWithFlatSpot', () {
      for (var af in allFloat32x4Functions()) {
        for (var o = 0.0; o <= 1.0; o += 0.05) {
          var scalar = af.derivativeWithFlatSpot(o);
          var simd = af.derivativeEntryWithFlatSpot(Float32x4.splat(o));

          expect(simd.x, closeTo(scalar, 1e-6), reason: '${af.name} at $o');
        }
      }
    });
  });

  group('ActivationFunctionLinear', () {
    var af = ActivationFunctionLinear();

    test('is the identity', () {
      for (var x in [-10.0, -1.0, 0.0, 1.0, 10.0]) {
        expect(af.activate(x), equals(x));
      }

      var e = af.activateEntry(Float32x4(1, 2, 3, 4));
      expect([e.x, e.y, e.z, e.w], equals([1.0, 2.0, 3.0, 4.0]));
    });

    test('derivative is 1 (plus the flat spot)', () {
      expect(af.derivative(0.5), equals(1.0));
      expect(af.derivativeWithFlatSpot(0.5), equals(1.0 + af.flatSpot));
      expect(af.derivativeEntry(Float32x4.splat(0.5)).x, equals(1.0));
      // `Float32x4` lanes are single precision:
      expect(
        af.derivativeEntryWithFlatSpot(Float32x4.splat(0.5)).x,
        closeTo(1.0 + af.flatSpot, 1e-6),
      );
    });

    test('scope is the input layer', () {
      expect(af.scope, equals([ActivationFunctionScope.input]));
    });
  });

  group('ActivationFunctionSigmoid', () {
    var af = ActivationFunctionSigmoid();

    test('reference values', () {
      expect(af.activate(0), equals(0.5));
      expect(af.activate(1), closeTo(0.7310585786300049, 1e-12));
      expect(af.activate(-1), closeTo(0.2689414213699951, 1e-12));
    });

    test('is monotonically increasing and bounded to 0..1', () {
      var previous = double.negativeInfinity;
      for (var x = -20.0; x <= 20.0; x += 0.5) {
        var y = af.activate(x);
        expect(y > previous, isTrue, reason: 'not increasing at $x');
        expect(y >= 0 && y <= 1, isTrue, reason: 'out of range at $x');
        previous = y;
      }
    });

    test('is symmetric around 0.5', () {
      for (var x = 0.0; x <= 10.0; x += 0.5) {
        expect(af.activate(x) + af.activate(-x), closeTo(1.0, 1e-12));
      }
    });

    test('derivative is o*(1-o), maximal at 0.5', () {
      expect(af.derivative(0.5), equals(0.25));
      expect(af.derivative(0.0), equals(0.0));
      expect(af.derivative(1.0), equals(0.0));
      expect(af.derivativeWithFlatSpot(0.0), equals(af.flatSpot));
    });

    test('scope is hidden and output layers', () {
      expect(
        af.scope,
        equals([
          ActivationFunctionScope.hidden,
          ActivationFunctionScope.output,
        ]),
      );
    });
  });

  group('ActivationFunctionSigmoidFast', () {
    var af = ActivationFunctionSigmoidFast();

    test('reference values', () {
      expect(af.activate(0), equals(0.5));
      expect(af.activate(-1), closeTo(0.2272727272727273, 1e-12));
      expect(af.activate(1), closeTo(1.0 - 0.2272727272727273, 1e-12));
    });

    test('is monotonically increasing and bounded to 0..1', () {
      var previous = double.negativeInfinity;
      for (var x = -50.0; x <= 50.0; x += 0.5) {
        var y = af.activate(x);
        expect(y > previous, isTrue, reason: 'not increasing at $x');
        expect(y >= 0 && y <= 1, isTrue, reason: 'out of range at $x');
        previous = y;
      }
    });

    test('approximates the real Sigmoid', () {
      var sigmoid = ActivationFunctionSigmoid();
      for (var x = -6.0; x <= 6.0; x += 0.5) {
        expect(
          (af.activate(x) - sigmoid.activate(x)).abs() < 0.15,
          isTrue,
          reason: 'too far from Sigmoid at $x',
        );
      }
    });
  });

  group('ActivationFunctionSigmoidBoundedFast', () {
    test('reaches exactly 0 and 1 at the limits', () {
      for (var scale in [2.0, 6.0, 10.0]) {
        var af = ActivationFunctionSigmoidBoundedFast(scale: scale);

        expect(af.lowerLimit, equals(-scale));
        expect(af.upperLimit, equals(scale));

        expect(af.activate(-scale), closeTo(0.0, 1e-12));
        expect(af.activate(0), equals(0.5));
        expect(af.activate(scale), closeTo(1.0, 1e-12));

        // Beyond the limits it saturates:
        expect(af.activate(-scale * 100), closeTo(0.0, 1e-12));
        expect(af.activate(scale * 100), closeTo(1.0, 1e-12));
      }
    });

    test('SIMD saturates at the limits too', () {
      for (var scale in [2.0, 6.0]) {
        var af = ActivationFunctionSigmoidBoundedFast(scale: scale);

        var low = af.activateEntry(Float32x4.splat(-scale * 100));
        var high = af.activateEntry(Float32x4.splat(scale * 100));

        expect(low.x, closeTo(0.0, 1e-6), reason: 'scale: $scale');
        expect(high.x, closeTo(1.0, 1e-6), reason: 'scale: $scale');
      }
    });

    test('the scale changes the shape of the curve', () {
      var narrow = ActivationFunctionSigmoidBoundedFast(scale: 2);
      var wide = ActivationFunctionSigmoidBoundedFast(scale: 10);

      // The narrower scale saturates faster:
      expect(narrow.activate(2) > wide.activate(2), isTrue);
      expect(
        narrow.activateEntry(Float32x4.splat(2)).x >
            wide.activateEntry(Float32x4.splat(2)).x,
        isTrue,
        reason: 'the SIMD version must honor the scale as well',
      );
    });

    test('is monotonically increasing and bounded to 0..1', () {
      var af = ActivationFunctionSigmoidBoundedFast(scale: 6);
      var previous = -1.0;
      for (var x = -6.0; x <= 6.0; x += 0.25) {
        var y = af.activate(x);
        expect(y >= previous, isTrue, reason: 'not increasing at $x');
        expect(y >= 0 && y <= 1, isTrue, reason: 'out of range at $x');
        previous = y;
      }
    });

    test('default initialWeightScale is 2', () {
      expect(
        ActivationFunctionSigmoidBoundedFast().initialWeightScale,
        equals(2.0),
      );
    });
  });

  group('ActivationFunctionSigmoidFastInt100', () {
    var af = ActivationFunctionSigmoidFastInt100();

    test('reference values', () {
      expect(af.activate(0), equals(50));
      expect(af.activate(-1), equals(34));
      expect(af.activate(1), equals(66));
      expect(af.activate(-100), equals(1));
      expect(af.activate(100), equals(99));
    });

    test('stays within 0..100', () {
      for (var x = -1000; x <= 1000; x += 7) {
        var y = af.activate(x);
        expect(y >= 0 && y <= 100, isTrue, reason: 'out of range at $x');
      }
    });

    test('activateEntry matches activate for every lane', () {
      var e = af.activateEntry(Int32x4(-10, -1, 1, 10));
      expect(e.x, equals(af.activate(-10)));
      expect(e.y, equals(af.activate(-1)));
      expect(e.z, equals(af.activate(1)));
      expect(e.w, equals(af.activate(10)));
    });

    test('derivativeEntry matches derivative for every lane', () {
      var e = af.derivativeEntry(Int32x4(10, 20, 30, 40));
      expect(e.x, equals(af.derivative(10)));
      expect(e.y, equals(af.derivative(20)));
      expect(e.z, equals(af.derivative(30)));
      expect(e.w, equals(af.derivative(40)));
    });
  });

  group('ActivationFunctionSigmoidFastInt', () {
    test('scaleCenter is half of scaleMax', () {
      expect(ActivationFunctionSigmoidFastInt(100).scaleCenter, equals(50));
      expect(ActivationFunctionSigmoidFastInt(200).scaleCenter, equals(100));
    });

    test('activate is centered and bounded by scaleMax', () {
      for (var scaleMax in [100, 200, 1000]) {
        var af = ActivationFunctionSigmoidFastInt(scaleMax);

        expect(af.activate(0), equals(scaleMax ~/ 2));

        for (var x = -500; x <= 500; x += 13) {
          var y = af.activate(x);
          expect(
            y >= 0 && y <= scaleMax,
            isTrue,
            reason: 'scaleMax:$scaleMax out of range at $x -> $y',
          );
        }
      }
    });

    test('derivativeEntry honors scaleMax', () {
      var af = ActivationFunctionSigmoidFastInt(200);

      expect(af.derivative(10), equals(10 * (200 - 10)));

      var e = af.derivativeEntry(Int32x4(10, 20, 30, 40));
      expect(e.x, equals(af.derivative(10)));
      expect(e.y, equals(af.derivative(20)));
      expect(e.z, equals(af.derivative(30)));
      expect(e.w, equals(af.derivative(40)));
    });

    test('activateEntry matches activate for every lane', () {
      var af = ActivationFunctionSigmoidFastInt(200);
      var e = af.activateEntry(Int32x4(-10, -1, 1, 10));

      expect(e.x, equals(af.activate(-10)));
      expect(e.y, equals(af.activate(-1)));
      expect(e.z, equals(af.activate(1)));
      expect(e.w, equals(af.activate(10)));
    });
  });

  group('ActivationFunction: random weights', () {
    test('createRandomWeight is bounded by initialWeightScale', () {
      var af = ActivationFunctionSigmoid(initialWeightScale: 4);
      var random = Random(1);

      for (var i = 0; i < 200; ++i) {
        var w = af.createRandomWeight(random);
        expect(w >= -4 && w <= 4, isTrue, reason: 'out of range: $w');
      }
    });

    test('createRandomWeight honors an explicit scale', () {
      var af = ActivationFunctionSigmoid(initialWeightScale: 1);
      var random = Random(1);

      var anyAboveOne = false;
      for (var i = 0; i < 200; ++i) {
        var w = af.createRandomWeight(random, scale: 100);
        expect(w >= -100 && w <= 100, isTrue);
        if (w.abs() > 1) anyAboveOne = true;
      }

      expect(anyAboveOne, isTrue, reason: 'the scale must be applied');
    });

    test('createRandomWeights honors an explicit scale', () {
      var af = ActivationFunctionSigmoid(initialWeightScale: 1);

      var weights = af.createRandomWeights(Random(1), 200, scale: 100);

      expect(weights.length, equals(200));
      expect(weights.every((w) => w >= -100 && w <= 100), isTrue);
      expect(
        weights.any((w) => w.abs() > 1),
        isTrue,
        reason: 'the scale must be forwarded to each weight',
      );
    });

    test('createRandomWeights defaults to initialWeightScale', () {
      var af = ActivationFunctionSigmoid(initialWeightScale: 2);
      var weights = af.createRandomWeights(Random(1), 100);

      expect(weights.every((w) => w >= -2 && w <= 2), isTrue);
    });

    test('is reproducible with the same seed', () {
      var af = ActivationFunctionSigmoid();
      expect(
        af.createRandomWeights(Random(42), 10),
        equals(af.createRandomWeights(Random(42), 10)),
      );
    });
  });

  group('ActivationFunction: byName', () {
    test('resolves every Float32x4 function', () {
      expect(
        ActivationFunction.byName('Linear'),
        isA<ActivationFunctionLinear>(),
      );
      expect(
        ActivationFunction.byName('Sigmoid'),
        isA<ActivationFunctionSigmoid>(),
      );
      expect(
        ActivationFunction.byName('SigmoidFast'),
        isA<ActivationFunctionSigmoidFast>(),
      );
      expect(
        ActivationFunction.byName('SigmoidBoundedFast'),
        isA<ActivationFunctionSigmoidBoundedFast>(),
      );
    });

    test('resolves the Int32x4 functions', () {
      var af100 = ActivationFunction.byName('SigmoidFastInt100');
      expect(af100, isA<ActivationFunctionSigmoidFastInt100>());
      expect(af100.name, equals('SigmoidFastInt100'));

      var af =
          ActivationFunction.byName('SigmoidFastInt', scaleMax: 200)
              as ActivationFunctionSigmoidFastInt;
      expect(af.scaleMax, equals(200));
    });

    test('applies initialWeightScale and scale', () {
      var af = ActivationFunction.byName('Sigmoid', initialWeightScale: 3);
      expect(af.initialWeightScale, equals(3.0));

      var bounded =
          ActivationFunction.byName('SigmoidBoundedFast', scale: 4)
              as ActivationFunctionSigmoidBoundedFast;
      expect(bounded.scale, equals(4.0));
    });

    test('unknown name throws', () {
      expect(
        () => ActivationFunction.byName('Nope'),
        throwsA(isA<StateError>()),
      );
    });
  });

  group('ActivationFunction: JSON', () {
    test('round-trips every function', () {
      var functions = <ActivationFunction>[
        ActivationFunctionLinear(initialWeightScale: 3),
        ActivationFunctionSigmoid(initialWeightScale: 2),
        ActivationFunctionSigmoidFast(),
        ActivationFunctionSigmoidBoundedFast(scale: 4, initialWeightScale: 5),
        ActivationFunctionSigmoidFastInt100(),
        ActivationFunctionSigmoidFastInt(200),
      ];

      for (var af in functions) {
        var decoded = ActivationFunction.fromJson(af.toJsonMap());

        expect(
          decoded.runtimeType,
          equals(af.runtimeType),
          reason: 'type of ${af.name}',
        );
        expect(decoded.name, equals(af.name));
        expect(
          decoded.initialWeightScale,
          equals(af.initialWeightScale),
          reason: 'initialWeightScale of ${af.name}',
        );
      }
    });

    test('keeps the scale of SigmoidBoundedFast', () {
      var af = ActivationFunctionSigmoidBoundedFast(scale: 4);
      expect(af.toJsonMap()['scale'], equals(4.0));

      var decoded =
          ActivationFunction.fromJson(af.toJsonMap())
              as ActivationFunctionSigmoidBoundedFast;

      expect(decoded.scale, equals(4.0));
      expect(decoded.activate(2), equals(af.activate(2)));
      expect(
        decoded.activateEntry(Float32x4.splat(2)).x,
        equals(af.activateEntry(Float32x4.splat(2)).x),
      );
    });

    test('keeps the scaleMax of SigmoidFastInt', () {
      var af = ActivationFunctionSigmoidFastInt(200);
      expect(af.toJsonMap()['scaleMax'], equals(200));

      var decoded =
          ActivationFunction.fromJson(af.toJsonMap())
              as ActivationFunctionSigmoidFastInt;

      expect(decoded.scaleMax, equals(200));
      expect(decoded.activate(10), equals(af.activate(10)));
    });

    test('decodes from an encoded String', () {
      var af = ActivationFunctionSigmoid(initialWeightScale: 2);
      var decoded = ActivationFunction.fromJson(af.toJson());

      expect(decoded.name, equals('Sigmoid'));
      expect(decoded.initialWeightScale, equals(2.0));

      expect(af.toJson(withIndent: true), contains('\n'));
    });

    test('toString is the runtime type', () {
      expect(
        ActivationFunctionSigmoid().toString(),
        equals('ActivationFunctionSigmoid'),
      );
    });
  });

  group('ActivationFunction: constants', () {
    test('TOO_SMALL/TOO_BIG', () {
      expect(ActivationFunction.TOO_SMALL, equals(-1.0E20));
      expect(ActivationFunction.TOO_BIG, equals(1.0E20));
    });

    test('default flatSpot and bias value', () {
      var af = ActivationFunctionSigmoid();
      expect(af.flatSpot, equals(0.0001));
      expect(af.initialWeightBiasValue, equals(0.0));
    });

    test('shared Float32x4 constants', () {
      expect(ActivationFunctionFloat32x4.entryOfZeroes.x, equals(0.0));
      expect(ActivationFunctionFloat32x4.entryOfOnes.x, equals(1.0));
      expect(ActivationFunctionFloat32x4.entryOfMinusOnes.x, equals(-1.0));
      expect(ActivationFunctionFloat32x4.entryOfHalf.x, equals(0.5));
      expect(ActivationFunctionFloat32x4.entryOfTwos.x, equals(2.0));
      expect(ActivationFunctionFloat32x4.entryOfTwosAndHalf.x, equals(2.5));
      expect(ActivationFunctionFloat32x4.entryOfThrees.x, equals(3.0));
    });
  });
}
