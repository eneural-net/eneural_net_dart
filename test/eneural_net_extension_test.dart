import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:test/test.dart';

void main() {
  group('Int32x4Extension', () {
    var e = Int32x4(1, 2, 3, 4);

    test('conversions', () {
      expect(e.toInts(), equals([1, 2, 3, 4]));

      var f = e.toFloat32x4();
      expect([f.x, f.y, f.z, f.w], equals([1.0, 2.0, 3.0, 4.0]));
    });

    test('filter/filterValues/filterToDoubleValues/map', () {
      expect(e.filter((v) => v + Int32x4(1, 1, 1, 1)).x, equals(2));
      expect(e.filterValues((v) => v * 2).toInts(), equals([2, 4, 6, 8]));

      var f = e.filterToDoubleValues((v) => v / 2);
      expect([f.x, f.y, f.z, f.w], equals([0.5, 1.0, 1.5, 2.0]));

      expect(e.map((v) => v.x + v.w), equals(5));
    });

    test('operators * and ~/', () {
      expect((e * Int32x4(2, 2, 2, 2)).toInts(), equals([2, 4, 6, 8]));
      expect((e ~/ Int32x4(2, 2, 2, 2)).toInts(), equals([0, 1, 1, 2]));
    });

    test('minInLane/maxInLane', () {
      expect(e.minInLane, equals(1));
      expect(e.maxInLane, equals(4));
      expect(Int32x4(9, 2, 8, 3).minInLane, equals(2));
      expect(Int32x4(9, 2, 8, 3).maxInLane, equals(9));
      expect(Int32x4(1, 9, 2, 3).maxInLane, equals(9));
      expect(Int32x4(1, 2, 9, 3).maxInLane, equals(9));
      expect(Int32x4(5, 4, 3, 1).minInLane, equals(1));
    });

    test('sumLane/sumLanePartial', () {
      expect(e.sumLane, equals(10));
      expect(e.sumLanePartial(1), equals(1));
      expect(e.sumLanePartial(2), equals(3));
      expect(e.sumLanePartial(3), equals(6));
      expect(e.sumLanePartial(4), equals(10));
      expect(() => e.sumLanePartial(0), throwsA(isA<StateError>()));
      expect(() => e.sumLanePartial(5), throwsA(isA<StateError>()));
    });

    test('sumSquaresLane/sumSquaresLanePartial', () {
      expect(e.sumSquaresLane, equals(30));
      expect(e.sumSquaresLanePartial(1), equals(1));
      expect(e.sumSquaresLanePartial(2), equals(5));
      expect(e.sumSquaresLanePartial(3), equals(14));
      expect(e.sumSquaresLanePartial(4), equals(30));
      expect(() => e.sumSquaresLanePartial(9), throwsA(isA<StateError>()));
    });

    test('equalsValues', () {
      expect(e.equalsValues(Int32x4(1, 2, 3, 4)), isTrue);
      expect(e.equalsValues(Int32x4(1, 2, 3, 5)), isFalse);
    });
  });

  group('Float32x4Extension', () {
    var e = Float32x4(1, 2, 3, 4);

    test('conversions', () {
      expect(e.toDoubles(), equals([1.0, 2.0, 3.0, 4.0]));
      expect(
        Float32x4(1.9, 2.9, 3.9, 4.9).toInt32x4().toInts(),
        equals([1, 2, 3, 4]),
      );
      expect(
        Float32x4(1.9, 2.9, 3.9, 4.9).toIntAsFloat32x4().toDoubles(),
        equals([1.0, 2.0, 3.0, 4.0]),
      );
    });

    test('filter/filterValues/filterToIntValues/map', () {
      expect(e.filter((v) => v + Float32x4.splat(1)).x, equals(2.0));
      expect(e.filterValues((v) => v * 2).toDoubles(), equals([2, 4, 6, 8]));
      expect(
        e.filterToIntValues((v) => (v * 10).toInt()).toInts(),
        equals([10, 20, 30, 40]),
      );
      expect(e.map((v) => v.x + v.w), equals(5.0));
    });

    test('minInLane/maxInLane', () {
      expect(e.minInLane, equals(1.0));
      expect(e.maxInLane, equals(4.0));
      expect(Float32x4(9, 2, 8, 3).minInLane, equals(2.0));
      expect(Float32x4(1, 9, 2, 3).maxInLane, equals(9.0));
      expect(Float32x4(1, 2, 9, 3).maxInLane, equals(9.0));
      expect(Float32x4(5, 4, 3, 1).minInLane, equals(1.0));
    });

    test('sumLane/sumLanePartial', () {
      expect(e.sumLane, equals(10.0));
      expect(e.sumLanePartial(1), equals(1.0));
      expect(e.sumLanePartial(2), equals(3.0));
      expect(e.sumLanePartial(3), equals(6.0));
      expect(e.sumLanePartial(4), equals(10.0));
      expect(() => e.sumLanePartial(5), throwsA(isA<StateError>()));
    });

    test('sumSquaresLane/sumSquaresLanePartial', () {
      expect(e.sumSquaresLane, equals(30.0));
      expect(e.sumSquaresLanePartial(1), equals(1.0));
      expect(e.sumSquaresLanePartial(2), equals(5.0));
      expect(e.sumSquaresLanePartial(3), equals(14.0));
      expect(e.sumSquaresLanePartial(4), equals(30.0));
      expect(() => e.sumSquaresLanePartial(0), throwsA(isA<StateError>()));
    });

    test('equalsValues', () {
      expect(e.equalsValues(Float32x4(1, 2, 3, 4)), isTrue);
      expect(e.equalsValues(Float32x4(1, 2, 3, 5)), isFalse);
    });
  });

  group('Equality implementations', () {
    test('Int32x4Equality', () {
      var eq = Int32x4Equality();

      expect(eq.equals(Int32x4(1, 2, 3, 4), Int32x4(1, 2, 3, 4)), isTrue);
      expect(eq.equals(Int32x4(1, 2, 3, 4), Int32x4(0, 2, 3, 4)), isFalse);
      expect(
        eq.hash(Int32x4(1, 2, 3, 4)),
        equals(eq.hash(Int32x4(1, 2, 3, 4))),
      );
      expect(eq.isValidKey(Int32x4(1, 2, 3, 4)), isTrue);
      expect(eq.isValidKey('nope'), isFalse);
    });

    test('Float32x4Equality', () {
      var eq = Float32x4Equality();

      expect(eq.equals(Float32x4(1, 2, 3, 4), Float32x4(1, 2, 3, 4)), isTrue);
      expect(eq.equals(Float32x4(1, 2, 3, 4), Float32x4(0, 2, 3, 4)), isFalse);
      expect(
        eq.hash(Float32x4(1, 2, 3, 4)),
        equals(eq.hash(Float32x4(1, 2, 3, 4))),
      );
      expect(eq.isValidKey(Float32x4(1, 2, 3, 4)), isTrue);
      expect(eq.isValidKey(1), isFalse);
    });
  });

  group('ListExtension', () {
    test('lastIndex/getReversed/getValueIfExists', () {
      var list = [1, 2, 3];

      expect(list.lastIndex, equals(2));
      expect(list.getReversed(0), equals(3));
      expect(list.getReversed(2), equals(1));

      expect(list.getValueIfExists(0), equals(1));
      expect(list.getValueIfExists(3), isNull);
      expect(list.getValueIfExists(-1), isNull);
    });

    test('setAllWithValue/setAllWith/setAllWithList', () {
      var list = [1, 2, 3];

      expect(list.setAllWithValue(9), equals(3));
      expect(list, equals([9, 9, 9]));

      expect(list.setAllWith((i, v) => i), equals(3));
      expect(list, equals([0, 1, 2]));

      expect(list.setAllWithList([7, 8, 9]), equals(3));
      expect(list, equals([7, 8, 9]));

      expect(list.setAllWithList([0, 0, 1, 2, 3], 2), equals(3));
      expect(list, equals([1, 2, 3]));
    });

    test('allEquals', () {
      expect([1, 1, 1].allEquals(1), isTrue);
      expect([1, 1, 2].allEquals(1), isFalse);
      expect(<int>[].allEquals(1), isTrue, reason: 'vacuously true');
    });

    test('toStringElements/computeHashcode', () {
      expect([1, 2].toStringElements(), equals(['1', '2']));
      expect([1, 2].computeHashcode(), equals([1, 2].computeHashcode()));
      expect([1, 2].computeHashcode(), isNot(equals([2, 1].computeHashcode())));
    });

    test('removeFromBegin/removeFromEnd', () {
      var list = [1, 2, 3, 4, 5];
      expect(list.removeFromBegin(2), equals(2));
      expect(list, equals([3, 4, 5]));

      expect(list.removeFromEnd(1), equals(1));
      expect(list, equals([3, 4]));

      expect(list.removeFromBegin(0), equals(0));
      expect(list.removeFromEnd(-1), equals(0));

      expect(list.removeFromBegin(100), equals(2), reason: 'clamped');
      expect(list, isEmpty);
    });

    test('removeFromEnd clamps to the length', () {
      var list = [1, 2, 3];
      expect(list.removeFromEnd(100), equals(3));
      expect(list, isEmpty);
    });

    test('ensureMaximumSize', () {
      var list = [1, 2, 3, 4, 5];
      expect(list.ensureMaximumSize(10), equals(0), reason: 'already small');
      expect(list, equals([1, 2, 3, 4, 5]));

      expect(list.ensureMaximumSize(3), equals(2));
      expect(list, equals([3, 4, 5]));

      var list2 = [1, 2, 3, 4, 5];
      expect(list2.ensureMaximumSize(3, removeFromEnd: true), equals(2));
      expect(list2, equals([1, 2, 3]));

      var list3 = [1, 2, 3, 4, 5];
      expect(list3.ensureMaximumSize(4, removeExtras: 2), equals(3));
      expect(list3, equals([4, 5]));
    });

    test('asDoubles/asInts', () {
      expect(<double>[1.0, 2.0].asDoubles(), equals([1.0, 2.0]));
      expect(<int>[1, 2].asInts(), equals([1, 2]));

      expect(<dynamic>['1', '2'].asDoubles(), equals([1.0, 2.0]));
      expect(<dynamic>['1', '2'].asInts(), equals([1, 2]));
      expect(<num>[1, 2].asDoubles(), equals([1.0, 2.0]));
    });
  });

  group('SetExtension', () {
    test('allEquals', () {
      expect({1}.allEquals(1), isTrue);
      expect({1, 2}.allEquals(1), isFalse);
      expect(<int>{}.allEquals(1), isTrue, reason: 'vacuously true');
    });

    test('toStringElements/computeHashcode', () {
      expect({1, 2}.toStringElements().toSet(), equals({'1', '2'}));
      expect({1, 2}.computeHashcode(), equals({2, 1}.computeHashcode()));
    });
  });

  group('IterableExtension', () {
    test('groupBy', () {
      var groups = [1, 2, 3, 4, 5, 6].groupBy((e) => e % 3);

      expect(groups.keys.toSet(), equals({0, 1, 2}));
      expect(groups[0], equals([3, 6]));
      expect(groups[1], equals([1, 4]));
      expect(groups[2], equals([2, 5]));
    });

    test('groupBy on an empty iterable', () {
      expect(<int>[].groupBy((e) => e), isEmpty);
    });
  });

  group('ListNumExtension', () {
    test('castElement', () {
      expect(<int>[1].castElement(2.9), equals(2));
      expect(<double>[1.0].castElement(2), equals(2.0));
    });

    test('mapToList/mapToSet/toInts/toDoubles/toStrings', () {
      var list = <num>[1, 2, 2];

      expect(list.mapToList((n) => n * 2), equals([2, 4, 4]));
      expect(list.mapToSet((n) => n * 2), equals({2, 4}));
      expect(list.toInts(), equals([1, 2, 2]));
      expect(list.toDoubles(), equals([1.0, 2.0, 2.0]));
      expect(list.toStrings(), equals(['1', '2', '2']));
    });

    test('sum/sumSquares/mean/squaresMean', () {
      var list = <num>[1, 2, 3, 4];

      expect(list.sum, equals(10));
      expect(list.sumSquares, equals(30));
      expect(list.mean, equals(2.5));
      expect(list.squaresMean, equals(7.5));
    });

    test('sum/sumSquares of an empty list', () {
      expect(<num>[].sum, equals(0));
      expect(<num>[].sumSquares, equals(0));
    });

    test('standardDeviation', () {
      // Classic example: mean 5, population standard deviation 2.
      expect(
        <num>[2, 4, 4, 4, 5, 5, 7, 9].standardDeviation,
        closeTo(2.0, 1e-12),
      );
      expect(<num>[5, 5, 5].standardDeviation, equals(0.0));
      expect(<num>[].standardDeviation, equals(0.0));
    });

    test('square/abs', () {
      expect(<num>[1, -2, 3].square, equals([1, 4, 9]));
      expect(<num>[1, -2, 3].abs, equals([1, 2, 3]));
    });

    test('movingAverage', () {
      expect(<num>[1, 2, 3, 4].movingAverage(2), equals([1.5, 2.5, 3.5]));
      expect(
        <num>[1, 2, 3, 4].movingAverage(4),
        equals([2.5]),
        reason: 'window >= length returns the mean',
      );
      expect(<num>[1, 2, 3, 4].movingAverage(10), equals([2.5]));
    });

    test('mergeBlocks', () {
      expect(<num>[1, 2, 3, 4].mergeBlocks(2), equals([1.5, 3.5]));
      expect(
        <num>[1, 2, 3, 4, 5].mergeBlocks(2),
        equals([1.5, 3.5, 5.0]),
        reason: 'the tail block can be shorter',
      );
      expect(<num>[1, 2].mergeBlocks(5), equals([1.5]));
    });

    test('diff/diffFromSignal', () {
      expect(<num>[3, 5].diff([1, 1]), equals([2, 4]));
      expect(
        <num>[3, 5].diffFromSignal(SignalFloat32x4.from([1, 1])),
        equals([2, 4]),
      );
    });

    test('operators', () {
      var a = <num>[4, 6];
      expect(a - [1, 2], equals([3, 4]));
      expect(a * [2, 2], equals([8, 12]));
      expect(a / [2, 2], equals([2.0, 3.0]));
      expect(a ~/ [3, 4], equals([1, 1]));
    });

    test('plus is the element-wise sum', () {
      var a = <num>[4, 6];

      expect(a.plus([1, 2]), equals([5, 8]));
      expect(ListNumExtension(a) + [1, 2], equals([5, 8]));

      // `List` defines `operator +` as concatenation and an instance member
      // wins over an extension member, so `a + b` does NOT sum:
      expect(a + [1, 2], equals([4, 6, 1, 2]));
    });

    test('statistics', () {
      var st = <num>[1, 2, 3, 4].statistics;
      expect(st.mean, equals(2.5));
      expect(st.min, equals(1));
      expect(st.max, equals(4));
    });
  });

  group('ListDoubleExtension', () {
    test('sum/sumSquares/mean/squaresMean', () {
      var list = <double>[1, 2, 3, 4];

      expect(list.sum, equals(10.0));
      expect(list.sumSquares, equals(30.0));
      expect(list.mean, equals(2.5));
      expect(list.squaresMean, equals(7.5));
      expect(list.castElement(2), equals(2.0));
      expect(list.toDoubles(), equals([1.0, 2.0, 3.0, 4.0]));
    });

    test('empty list', () {
      expect(<double>[].sum, equals(0.0));
      expect(<double>[].sumSquares, equals(0.0));
      expect(<double>[].standardDeviation, equals(0.0));
    });

    test('standardDeviation', () {
      expect(
        <double>[2, 4, 4, 4, 5, 5, 7, 9].standardDeviation,
        closeTo(2.0, 1e-12),
      );
    });

    test('square/abs/diff/diffFromSignal', () {
      expect(<double>[1, -2].square, equals([1.0, 4.0]));
      expect(<double>[1, -2].abs, equals([1.0, 2.0]));
      expect(<double>[3, 5].diff([1, 1]), equals([2.0, 4.0]));
      expect(
        <double>[3, 5].diffFromSignal(SignalFloat32x4.from([1, 1])),
        equals([2.0, 4.0]),
      );
    });

    test('operators', () {
      var a = <double>[4, 6];
      expect(a - [1, 2], equals([3.0, 4.0]));
      expect(a * [2, 2], equals([8.0, 12.0]));
      expect(a / [2, 2], equals([2.0, 3.0]));
      expect(a ~/ [3, 4], equals([1, 1]));
    });

    test('plus is the element-wise sum', () {
      var a = <double>[4, 6];

      expect(a.plus([1, 2]), equals([5.0, 8.0]));
      expect(ListDoubleExtension(a) + [1, 2], equals([5.0, 8.0]));

      // Shadowed by `List.operator +` (concatenation):
      expect(a + [1, 2], equals([4.0, 6.0, 1.0, 2.0]));
    });

    test('statistics/statisticsWithSeries', () {
      var list = <double>[1, 2, 3, 4];

      expect(list.statistics.mean, equals(2.5));
      expect(list.statistics.series, isNull);
      expect(list.statisticsWithSeries.series, equals(list));
    });
  });

  group('ListIntExtension', () {
    test('sum/sumSquares/mean', () {
      var list = <int>[1, 2, 3, 4];

      expect(list.sum, equals(10));
      expect(list.sumSquares, equals(30));
      expect(list.mean, equals(2.5));
      expect(list.castElement(2.9), equals(2));
      expect(list.toInts(), equals([1, 2, 3, 4]));
    });

    test('empty list', () {
      expect(<int>[].sum, equals(0));
      expect(<int>[].sumSquares, equals(0));
      expect(<int>[].standardDeviation, equals(0));
    });

    test('standardDeviation', () {
      expect(
        <int>[2, 4, 4, 4, 5, 5, 7, 9].standardDeviation,
        closeTo(2.0, 1e-12),
      );
    });

    test('square/abs', () {
      expect(<int>[1, -2].square, equals([1, 4]));
      expect(<int>[1, -2].abs, equals([1, 2]));
    });

    test('statistics/statisticsWithSeries', () {
      var list = <int>[1, 2, 3, 4];
      expect(list.statistics.mean, equals(2.5));
      expect(list.statisticsWithSeries.series, equals(list));
    });
  });

  group('NumExtension', () {
    test('square/squareRoot/naturalExponent', () {
      num n = 3;
      expect(n.square, equals(9));
      expect((16 as num).squareRoot, equals(4.0));
      expect((0 as num).naturalExponent, equals(1.0));
      expect((1 as num).naturalExponent, closeTo(e, 1e-12));
    });

    test('clamp', () {
      expect((5 as num).clamp(0, 10), equals(5));
      expect((-5 as num).clamp(0, 10), equals(0));
      expect((50 as num).clamp(0, 10), equals(10));
    });

    test('signWithZeroTolerance', () {
      expect((5 as num).signWithZeroTolerance(), equals(1));
      expect((-5 as num).signWithZeroTolerance(), equals(-1));
      expect((0 as num).signWithZeroTolerance(), equals(0));
      expect((1.0e-30 as num).signWithZeroTolerance(), equals(0));
      expect((-1.0e-30 as num).signWithZeroTolerance(), equals(0));
      expect((0.5 as num).signWithZeroTolerance(1.0), equals(0));
    });
  });

  group('DoubleExtension', () {
    test('square/clamp', () {
      expect(3.0.square, equals(9.0));
      expect(5.0.clamp(0.0, 10.0), equals(5.0));
      expect((-5.0).clamp(0.0, 10.0), equals(0.0));
      expect(50.0.clamp(0.0, 10.0), equals(10.0));
    });

    test('signWithZeroTolerance', () {
      expect(5.0.signWithZeroTolerance(), equals(1));
      expect((-5.0).signWithZeroTolerance(), equals(-1));
      expect(0.0.signWithZeroTolerance(), equals(0));
      expect(1.0e-20.signWithZeroTolerance(), equals(0));
      expect((-1.0e-20).signWithZeroTolerance(), equals(0));
    });
  });

  group('IntExtension', () {
    test('square/clamp', () {
      expect(3.square, equals(9));
      expect(5.clamp(0, 10), equals(5));
      expect((-5).clamp(0, 10), equals(0));
      expect(50.clamp(0, 10), equals(10));
    });
  });

  group('DurationExtension', () {
    test('toStringUnit picks the largest unit', () {
      expect(Duration(days: 2).toStringUnit(), equals('2 d'));
      expect(Duration(hours: 5).toStringUnit(), equals('5 h'));
      expect(Duration(minutes: 5).toStringUnit(), equals('5 min'));
      expect(Duration(seconds: 5).toStringUnit(), equals('5 sec'));
      expect(Duration(milliseconds: 5).toStringUnit(), equals('5 ms'));
      expect(Duration(microseconds: 5).toStringUnit(), equals('5 μs'));
      expect(Duration.zero.toStringUnit(), equals(Duration.zero.toString()));
    });

    test('units can be disabled', () {
      expect(Duration(days: 2).toStringUnit(days: false), equals('48 h'));
      expect(Duration(hours: 2).toStringUnit(hours: false), equals('120 min'));
      expect(
        Duration(seconds: 2).toStringUnit(seconds: false),
        equals('2000 ms'),
      );
      expect(
        Duration(milliseconds: 2).toStringUnit(milliseconds: false),
        equals('2000 μs'),
      );
    });
  });
}
