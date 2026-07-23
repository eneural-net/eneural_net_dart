import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';
import 'package:test/test.dart';

void main() {
  group('Chronometer', () {
    test('starts and stops', () {
      var c = Chronometer('test');

      expect(c.name, equals('test'));
      expect(c.startTime, isNull);
      expect(c.stopTime, isNull);
      expect(c.elapsedTimeMs, equals(0), reason: 'not started yet');

      expect(c.start(), same(c));
      expect(c.startTime, isNotNull);
      expect(c.elapsedTimeMs, equals(0), reason: 'not stopped yet');

      expect(c.stop(), same(c));
      expect(c.stopTime, isNotNull);
      expect(c.elapsedTimeMs >= 0, isTrue);
    });

    test('default name', () {
      expect(Chronometer().name, equals('Chronometer'));
    });

    test('records operations and failures', () {
      var c = Chronometer('ops')..start();
      c.stop(operations: 1000, failedOperations: 7);

      expect(c.operations, equals(1000));
      expect(c.failedOperations, equals(7));
      expect(c.operationsAsString, isNotEmpty);
      expect(c.failedOperationsAsString, isNotEmpty);
    });

    test('elapsedTime conversions agree', () {
      var c = Chronometer()..start();
      c.stop();

      expect(c.elapsedTimeSec, equals(c.elapsedTimeMs / 1000));
      expect(c.elapsedTime, equals(Duration(milliseconds: c.elapsedTimeMs)));
    });

    test('hertz uses the elapsed time', () {
      var start = DateTime.now();
      var c = Chronometer()..start();
      while (DateTime.now().difference(start).inMilliseconds < 5) {}
      c.stop(operations: 1000);

      expect(c.hertz > 0, isTrue);
      expect(c.hertz, equals(c.computeHertz(1000)));
      expect(c.computeHertz(2000), equals(c.hertz * 2));
      expect(c.hertzAsString, endsWith('Hz'));
    });

    test('reset clears the whole state', () {
      var c = Chronometer('r')..start();
      c.stop(operations: 10, failedOperations: 5);

      c.reset();

      expect(c.startTime, isNull);
      expect(c.stopTime, isNull);
      expect(c.operations, equals(0));
      expect(c.failedOperations, equals(0));
      expect(c.elapsedTimeMs, equals(0));
    });

    test('operator + merges two chronometers', () {
      var c1 = Chronometer('a')..start();
      c1.stop(operations: 10, failedOperations: 1);

      var c2 = Chronometer('b')..start();
      c2.stop(operations: 20, failedOperations: 2);

      var sum = c1 + c2;

      expect(sum.name, equals('a'));
      expect(sum.operations, equals(30));
      expect(sum.failedOperations, equals(3));
      expect(sum.startTime, equals(c1.startTime));
      expect(sum.stopTime, isNotNull);
    });

    test('operator + with unfinished chronometers', () {
      var stopped = Chronometer('stopped')..start();
      stopped.stop(operations: 5);

      var running = Chronometer('running')..start();

      expect((stopped + running).stopTime, equals(stopped.stopTime));
      expect((running + stopped).stopTime, equals(stopped.stopTime));

      var neither = Chronometer('x') + Chronometer('y');
      expect(neither.stopTime, isNull);
      expect(neither.elapsedTimeMs, equals(0));
    });

    test('compareTo sorts by hertz', () {
      var slow = Chronometer('slow')..start();
      slow.stop(operations: 1);

      var fast = Chronometer('fast')..start();
      fast.stop(operations: 1000000);

      var list = [slow, fast]..sort();
      expect(list.first.hertz <= list.last.hertz, isTrue);
      expect(slow.compareTo(slow), equals(0));
    });

    test('toString contains the relevant info', () {
      var c = Chronometer('bench')..start();
      c.stop(operations: 100);

      var str = c.toString();
      expect(str, startsWith('bench{'));
      expect(str, contains('elapsedTime'));
      expect(str, contains('hertz'));
      expect(str, contains('ops'));
      expect(str, isNot(contains('fails')));

      c.failedOperations = 3;
      expect(c.toString(), contains('fails'));
    });

    test('formats large numbers', () {
      var c = Chronometer('big')..start();
      c.stop(operations: 1234567);
      expect(c.operationsAsString, contains(','));
    });
  });

  group('DataStatistics: compute', () {
    test('of a regular series', () {
      var st = DataStatistics.compute(<double>[1, 2, 3, 4, 5]);

      expect(st.length, equals(5));
      expect(st.min, equals(1.0));
      expect(st.max, equals(5.0));
      expect(st.center, equals(3.0));
      expect(st.sum, equals(15.0));
      expect(st.squaresSum, equals(55.0));
      expect(st.mean, equals(3.0));
      expect(st.squaresMean, equals(11.0));
    });

    test('sorts the series before computing min/max/center', () {
      var st = DataStatistics.compute(<double>[5, 1, 3, 2, 4]);

      expect(st.min, equals(1.0));
      expect(st.max, equals(5.0));
      expect(st.center, equals(3.0));
    });

    test('standardDeviation is the population standard deviation', () {
      // Classic example: mean 5, standard deviation 2.
      var list = <double>[2, 4, 4, 4, 5, 5, 7, 9];
      var st = DataStatistics.compute(list);

      expect(st.mean, equals(5.0));
      expect(st.standardDeviation, closeTo(2.0, 1e-12));

      // Consistent with the `List` extension:
      expect(st.standardDeviation, closeTo(list.standardDeviation, 1e-12));
    });

    test('standardDeviation of a constant series is zero', () {
      var st = DataStatistics.compute(<double>[7, 7, 7, 7]);
      expect(st.mean, equals(7.0));
      expect(st.standardDeviation, closeTo(0.0, 1e-12));
    });

    test('of an empty series', () {
      var st = DataStatistics.compute(<double>[]);

      expect(st.length, equals(0));
      expect(st.mean, equals(0.0), reason: 'must not be NaN');
      expect(st.standardDeviation, equals(0.0), reason: 'must not be NaN');
      expect(st.squaresMean, equals(0.0), reason: 'must not be NaN');
      expect(st.toString(), equals('{empty}'));
    });

    test('of a single element', () {
      var st = DataStatistics.compute(<double>[7]);

      expect(st.length, equals(1));
      expect(st.min, equals(7.0));
      expect(st.max, equals(7.0));
      expect(st.center, equals(7.0));
      expect(st.mean, equals(7.0));
      expect(st.sum, equals(7.0));
      expect(st.squaresSum, equals(49.0));
      expect(st.standardDeviation, equals(0.0));
    });

    test('of an int series', () {
      var st = DataStatistics.compute(<int>[1, 2, 3, 4]);
      expect(st.min, equals(1));
      expect(st.mean, equals(2.5));
    });

    test('lower and upper statistics', () {
      var st = DataStatistics.compute(<double>[1, 2, 3, 4]);

      expect(st.lowerStatistics, isNotNull);
      expect(st.upperStatistics, isNotNull);
      expect(st.lowerStatistics!.max, equals(2.0));
      expect(st.upperStatistics!.min, equals(3.0));

      // Not computed recursively:
      expect(st.lowerStatistics!.lowerStatistics, isNull);
    });

    test('can skip the lower/upper statistics', () {
      var st = DataStatistics.compute(<double>[
        1,
        2,
        3,
        4,
      ], computeLowerAndUpper: false);

      expect(st.lowerStatistics, isNull);
      expect(st.upperStatistics, isNull);
    });

    test('keepSeries', () {
      var list = <double>[1, 2, 3, 4];

      expect(DataStatistics.compute(list).series, isNull);
      expect(
        DataStatistics.compute(list, keepSeries: true).series,
        equals(list),
      );
    });
  });

  group('DataStatistics: construction', () {
    test('derives sum from mean and vice-versa', () {
      var fromMean = DataStatistics(
        4,
        1,
        4,
        2,
        mean: 2.5,
        standardDeviation: 0,
      );
      expect(fromMean.sum, equals(10.0));

      var fromSum = DataStatistics(4, 1, 4, 2, sum: 10, standardDeviation: 0);
      expect(fromSum.mean, equals(2.5));
    });

    test('derives standardDeviation from squaresSum and mean', () {
      // [2,4,4,4,5,5,7,9] -> sum:40, squaresSum:232, mean:5, stdDev:2
      var st = DataStatistics(8, 2, 9, 5, sum: 40, squaresSum: 232);

      expect(st.mean, equals(5.0));
      expect(st.standardDeviation, closeTo(2.0, 1e-12));
    });

    test('derives squaresSum from standardDeviation and mean', () {
      var st = DataStatistics(8, 2, 9, 5, mean: 5, standardDeviation: 2);
      expect(st.squaresSum, closeTo(232.0, 1e-9));
    });

    test('computeStandardDeviation/computeSquaresSum are inverses', () {
      var sd = DataStatistics.computeStandardDeviation(232, 5, 8);
      expect(sd, closeTo(2.0, 1e-12));

      var squaresSum = DataStatistics.computeSquaresSum(2, 5, 8);
      expect(squaresSum, closeTo(232.0, 1e-9));
    });

    test('computeStandardDeviation guards against invalid input', () {
      expect(DataStatistics.computeStandardDeviation(10, 1, 0), equals(0.0));
      // A negative variance (floating point noise) yields 0, never NaN:
      expect(DataStatistics.computeStandardDeviation(1, 10, 1), equals(0.0));
    });
  });

  group('DataStatistics: operations', () {
    test('isMeanInRange', () {
      var st = DataStatistics.compute(<double>[1, 2, 3, 4, 5]);

      expect(st.isMeanInRange(2.0, 4.0), isTrue);
      expect(st.isMeanInRange(4.0, 5.0), isFalse);
      expect(st.isMeanInRange(0.0, 10.0, 0.0, 10.0), isTrue);
      expect(st.isMeanInRange(0.0, 10.0, 100.0, 200.0), isFalse);
    });

    test('operator + merges two series', () {
      var a = DataStatistics.compute(<double>[1, 2, 3]);
      var b = DataStatistics.compute(<double>[4, 5, 6]);
      var merged = a + b;

      var whole = DataStatistics.compute(<double>[1, 2, 3, 4, 5, 6]);

      expect(merged.length, equals(6));
      expect(merged.min, equals(1.0));
      expect(merged.max, equals(6.0));
      expect(merged.sum, equals(21.0));
      expect(merged.mean, closeTo(whole.mean, 1e-12));
      expect(
        merged.standardDeviation,
        closeTo(whole.standardDeviation, 1e-12),
        reason: 'merging must yield the statistics of the whole series',
      );
    });

    test('operator / builds a ratio', () {
      var a = DataStatistics.compute(<double>[2, 4, 6]);
      var b = DataStatistics.compute(<double>[1, 2, 3]);
      var ratio = a / b;

      expect(ratio.length, equals(1.0));
      expect(ratio.mean, closeTo(2.0, 1e-12));
      expect(ratio.min, closeTo(2.0, 1e-12));
      expect(ratio.max, closeTo(2.0, 1e-12));
    });

    test('toString', () {
      var st = DataStatistics.compute(<double>[1, 2, 3, 4, 5]);

      expect(st.toString(), contains('#5'));
      expect(st.toString(precision: 0), isNotEmpty);
    });

    test('getDataFields/getDataValues/getDataMap', () {
      var st = DataStatistics.compute(<double>[1, 2, 3, 4]);

      expect(
        st.getDataFields(),
        equals(['mean', 'standardDeviation', 'length', 'min', 'max']),
      );
      expect(st.getDataValues().length, equals(5));

      var map = st.getDataMap();
      expect(map['mean'], equals(2.5));
      expect(map['min'], equals(1.0));
      expect(map['max'], equals(4.0));
      expect(map['length'], equals(4.0));
    });
  });

  group('DataEntryExtension.generateCSV', () {
    test('generates a header and one line per entry', () {
      var entries = [
        DataStatistics.compute(<double>[1, 2, 3]),
        DataStatistics.compute(<double>[4, 5, 6]),
      ];

      var csv = entries.generateCSV();
      var lines = csv.trim().split('\n');

      expect(lines.length, equals(3));
      expect(lines[0], equals('mean,standardDeviation,length,min,max'));
      expect(lines[1].split(',').length, equals(5));
    });

    test('accepts a custom separator and field names', () {
      var entries = [
        DataStatistics.compute(<double>[1, 2, 3]),
      ];

      var csv = entries.generateCSV(separator: ';', fieldsNames: ['a', 'b']);
      expect(csv.split('\n')[0], equals('a;b'));
    });

    test('of an empty list', () {
      expect(<DataStatistics>[].generateCSV(), equals(''));
    });
  });

  group('SeriesMapExtension.generateCSV', () {
    test('generates a column per series', () {
      var series = <String, List<double>?>{
        'a': [1, 2, 3],
        'b': [4, 5, 6],
      };

      var csv = series.generateCSV();
      var lines = csv.trim().split('\n');

      expect(lines.length, equals(4));
      expect(lines[0], equals('#,a,b'));
      expect(lines[1], equals('1,1.0,4.0'));
      expect(lines[3], equals('3,3.0,6.0'));
    });

    test('pads shorter series with the null value', () {
      var series = <String, List<double>?>{
        'a': [1, 2, 3],
        'b': [4],
        'c': null,
      };

      var csv = series.generateCSV(nullValue: -1);
      var lines = csv.trim().split('\n');

      expect(lines[1], equals('1,1.0,4.0,-1.0'));
      expect(lines[2], equals('2,2.0,-1.0,-1.0'));
    });

    test('accepts a custom separator and first index', () {
      var series = <String, List<int>?>{
        'a': [1, 2],
      };

      var csv = series.generateCSV(separator: ';', firstEntryIndex: 0);
      var lines = csv.trim().split('\n');

      expect(lines[0], equals('#;a'));
      expect(lines[1], equals('0;1'));
    });

    test('of an empty map', () {
      expect(<String, List<double>?>{}.generateCSV(), equals(''));
    });

    test('csvFileName', () {
      var name = <String, List<double>?>{}.csvFileName('prefix', 'name');

      expect(name, startsWith('prefix--name--'));
      expect(name, endsWith('.csv'));
    });
  });
}
