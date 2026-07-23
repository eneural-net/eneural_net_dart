import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  group('ScaleInt', () {
    test('basic properties', () {
      var scale = ScaleInt(0, 100);

      expect(scale.minValue, equals(0));
      expect(scale.maxValue, equals(100));
      expect(scale.range, equals(100));
      expect(scale.zero, equals(0));
      expect(scale.format, equals('int'));
      expect(scale.toN(1.9), equals(1));
      expect(scale.toString(), equals('ScaleInt{0 .. 100}'));
    });

    test('normalize/denormalize are integer operations', () {
      var scale = ScaleInt(0, 100);

      // Integer division: everything below `range` normalizes to 0.
      expect(scale.normalize(99), equals(0));
      expect(scale.normalize(100), equals(1));
      expect(scale.normalize(250), equals(2));
      expect(scale.normalizeNum(250.9), equals(2));

      expect(scale.denormalize(0), equals(0));
      expect(scale.denormalize(1), equals(100));
      expect(scale.denormalize(2), equals(200));
    });

    test('normalize with offset min', () {
      var scale = ScaleInt(10, 20);
      expect(scale.range, equals(10));
      expect(scale.normalize(10), equals(0));
      expect(scale.normalize(20), equals(1));
      expect(scale.denormalize(1), equals(20));
    });

    test('normalizeList/denormalizeList', () {
      var scale = ScaleInt(0, 10);
      expect(scale.normalizeList([0, 10, 20, 30]), equals([0, 1, 2, 3]));
      expect(scale.denormalizeList([0, 1, 2, 3]), equals([0, 10, 20, 30]));
    });
  });

  group('ScaleDouble', () {
    test('basic properties', () {
      var scale = ScaleDouble(0, 100);

      expect(scale.range, equals(100.0));
      expect(scale.zero, equals(0.0));
      expect(scale.format, equals('double'));
      expect(scale.toN(1), equals(1.0));
      expect(scale.toN(1), isA<double>());
      expect(scale.toString(), equals('ScaleDouble{0.0 .. 100.0}'));
    });

    test('normalize is the inverse of denormalize', () {
      var scale = ScaleDouble(-50, 150);

      for (var v in [-50.0, -10.0, 0.0, 33.3, 100.0, 150.0]) {
        var n = scale.normalize(v);
        expect(scale.denormalize(n), closeTo(v, 1e-12), reason: 'value: $v');
      }
    });

    test('normalizeNum accepts int', () {
      var scale = ScaleDouble(0, 10);
      expect(scale.normalizeNum(5), equals(0.5));
    });

    test('normalizeList/denormalizeList', () {
      var scale = ScaleDouble(0, 10);
      expect(scale.normalizeList([0, 5, 10]), equals([0.0, 0.5, 1.0]));
      expect(scale.denormalizeList([0, 0.5, 1]), equals([0.0, 5.0, 10.0]));
    });
  });

  group('ScaleZoomable', () {
    test('ScaleZoomableInt', () {
      var scale = ScaleZoomableInt(0, 100, 10);

      expect(scale.zoom, equals(10));
      expect(scale.rangeZoomed, equals(10));
      expect(scale.format, equals('ZoomableInt'));
      expect(scale.zero, equals(0));
      expect(scale.toN(2.9), equals(2));
      expect(scale.toString(), equals('ScaleZoomableInt{0 .. 100 * 10}'));

      expect(scale.normalize(100), equals(10));
      expect(scale.normalizeNum(100.9), equals(10));
      expect(scale.denormalize(10), equals(100));
    });

    test('ScaleZoomableDouble', () {
      var scale = ScaleZoomableDouble(0, 100, 10);

      expect(scale.zoom, equals(10.0));
      expect(scale.rangeZoomed, equals(10.0));
      expect(scale.format, equals('ZoomableDouble'));
      expect(
        scale.toString(),
        equals('ScaleZoomableDouble{0.0 .. 100.0 * 10.0}'),
      );

      expect(scale.normalize(100), equals(10.0));
      expect(scale.normalizeNum(100), equals(10.0));
      expect(scale.denormalize(10), equals(100.0));
    });
  });

  group('Scale: validation', () {
    test('rejects max <= min', () {
      expect(() => ScaleDouble(1, 1), throwsA(isA<ArgumentError>()));
      expect(() => ScaleDouble(2, 1), throwsA(isA<ArgumentError>()));
      expect(() => ScaleInt(0, 0), throwsA(isA<ArgumentError>()));
      expect(() => ScaleZoomableInt(5, 1, 2), throwsA(isA<ArgumentError>()));
    });
  });

  group('Scale: equality', () {
    test('same type and range are equal', () {
      expect(ScaleInt(0, 10), equals(ScaleInt(0, 10)));
      expect(ScaleInt(0, 10).hashCode, equals(ScaleInt(0, 10).hashCode));
      expect(ScaleDouble(0, 10), equals(ScaleDouble(0, 10)));
    });

    test('different type or range are not equal', () {
      expect(ScaleInt(0, 10), isNot(equals(ScaleInt(0, 20))));
      expect(ScaleDouble(0, 10), isNot(equals(ScaleInt(0, 10))));
      expect(ScaleDouble(0, 10), isNot(equals('not a scale')));
    });

    test('identical instance is equal', () {
      var scale = ScaleDouble.ZERO_TO_ONE;
      expect(scale, equals(scale));
    });
  });

  group('Scale: JSON', () {
    test('ScaleInt round-trip', () {
      var scale = ScaleInt(-5, 25);
      expect(
        scale.toJsonMap(),
        equals({'format': 'int', 'min': -5, 'max': 25}),
      );

      var decoded = Scale.fromJson(scale.toJsonMap());
      expect(decoded, equals(scale));
      expect(decoded, isA<ScaleInt>());
    });

    test('ScaleDouble round-trip', () {
      var scale = ScaleDouble(-5, 25);
      expect(
        scale.toJsonMap(),
        equals({'format': 'double', 'min': -5.0, 'max': 25.0}),
      );

      var decoded = Scale.fromJson(scale.toJsonMap());
      expect(decoded, equals(scale));
      expect(decoded, isA<ScaleDouble>());
    });

    test('ScaleZoomableDouble round-trip keeps zoom', () {
      var scale = ScaleZoomableDouble(0, 100, 10);
      expect(scale.toJsonMap()['zoom'], equals(10.0));

      var decoded = Scale.fromJson(scale.toJsonMap()) as ScaleZoomableDouble;
      expect(decoded, equals(scale));
      expect(decoded.zoom, equals(10.0));
      expect(decoded.rangeZoomed, equals(scale.rangeZoomed));
    });

    test('ScaleZoomableInt round-trip keeps zoom', () {
      var scale = ScaleZoomableInt(0, 100, 10);

      // The `zoom` must be serialized, otherwise it can't be restored:
      expect(scale.toJsonMap()['zoom'], equals(10));
      expect(scale.toJsonMap()['format'], equals('ZoomableInt'));

      var decoded = Scale.fromJson(scale.toJsonMap()) as ScaleZoomableInt;
      expect(decoded, equals(scale));
      expect(decoded.zoom, equals(10));
      expect(decoded.rangeZoomed, equals(scale.rangeZoomed));
    });

    test('accepts the legacy `ScaleZoomableInt` format name', () {
      var decoded =
          Scale.fromJson({
                'format': 'ScaleZoomableInt',
                'min': 0,
                'max': 100,
                'zoom': 10,
              })
              as ScaleZoomableInt;

      expect(decoded, equals(ScaleZoomableInt(0, 100, 10)));
    });

    test('toJson encodes a parseable String', () {
      var scale = ScaleDouble(0, 1);
      var json = scale.toJson();
      expect(json, contains('"format"'));
      expect(Scale.fromJson(json), equals(scale));

      expect(scale.toJson(withIndent: true), contains('\n'));
    });

    test('unknown format throws', () {
      expect(
        () => Scale.fromJson({'format': 'nope', 'min': 0, 'max': 1}),
        throwsA(isA<StateError>()),
      );
    });
  });
}
