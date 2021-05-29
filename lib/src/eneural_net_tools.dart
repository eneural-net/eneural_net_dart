import 'dart:math' as math;

import 'package:intl/intl.dart';
import 'package:swiss_knife/swiss_knife.dart';

import 'eneural_net_extension.dart';

/// A Chronometer useful for benchmarks.
class Chronometer implements Comparable<Chronometer> {
  /// The name/title of this chronometer.
  String name;

  Chronometer([this.name = 'Chronometer']);

  Chronometer._(this.name, this.operations, this.failedOperations,
      this._startTime, this._stopTime);

  DateTime? _startTime;

  /// The start [DateTime] of this chronometer.
  DateTime? get startTime => _startTime;

  /// Starts the chronometer.
  Chronometer start() {
    _startTime = DateTime.now();
    return this;
  }

  DateTime? _stopTime;

  /// The stop [DateTime] of this chronometer.
  DateTime? get stopTime => _stopTime;

  /// Stops the chronometer.
  Chronometer stop({int? operations, int? failedOperations}) {
    _stopTime = DateTime.now();

    if (operations != null) {
      this.operations = operations;
    }

    if (failedOperations != null) {
      this.failedOperations = failedOperations;
    }

    return this;
  }

  /// Elapsed time in milliseconds ([stopTime] - [startTime]).
  int get elapsedTimeMs => (_stopTime == null || _startTime == null)
      ? 0
      : (_stopTime!.millisecondsSinceEpoch -
          _startTime!.millisecondsSinceEpoch);

  /// Elapsed time in seconds ([stopTime] - [startTime]).
  double get elapsedTimeSec => elapsedTimeMs / 1000;

  /// Elapsed time ([stopTime] - [startTime]).
  Duration get elapsedTime => Duration(milliseconds: elapsedTimeMs);

  /// Operations performed while this chronometer was running.
  /// Used to compute [hertz].
  int operations = 0;

  /// Failed operations performed while this chronometer was running.
  int failedOperations = 0;

  /// Returns the [operations] hertz:
  /// The average operations per second of
  /// the period ([elapsedTimeSec]) of this chronometer.
  double get hertz => computeHertz(operations);

  String get hertzAsString => '${_formatNumber(hertz)} Hz';

  String get operationsAsString => _formatNumber(operations);

  String get failedOperationsAsString => _formatNumber(failedOperations);

  static final NumberFormat _numberFormatDecimal =
      NumberFormat.decimalPattern('en_US');

  String _formatNumber(num n) {
    var s = n.isFinite && n > 10000
        ? _numberFormatDecimal.format(n.toInt())
        : _numberFormatDecimal.format(n);
    return s;
  }

  /// Computes hertz for n [operations].
  double computeHertz(int operations) {
    return operations / elapsedTimeSec;
  }

  /// Resets this chronometer for a future [start] and [stop].
  void reset() {
    _startTime = null;
    _stopTime = null;
    operations = 0;
  }

  /// Returns a [String] with information of this chronometer:
  /// Example:
  /// ```
  ///   Backpropagation{elapsedTime: 2955 ms, hertz: 2030456.8527918782 Hz, ops: 6000000, startTime: 2021-04-30 22:16:54.437147, stopTime: 2021-04-30 22:16:57.392758}
  /// ```
  @override
  String toString() {
    return '$name{ elapsedTime: $elapsedTimeMs ms'
        ', hertz: $hertzAsString'
        ', ops: $operationsAsString${failedOperations != 0 ? ' (fails: $failedOperationsAsString)' : ''}'
        ', startTime: $_startTime .. +${elapsedTime.toStringUnit()} }';
  }

  Chronometer operator +(Chronometer other) {
    DateTime? end;
    if (_stopTime != null && other._stopTime != null) {
      end = _stopTime!.add(other.elapsedTime);
    } else if (_stopTime != null) {
      end = _stopTime;
    } else if (other._stopTime != null) {
      end = other._stopTime;
    }

    return Chronometer._(name, operations + other.operations,
        failedOperations + other.failedOperations, _startTime, end);
  }

  @override
  int compareTo(Chronometer other) => hertz.compareTo(other.hertz);
}

class DataStatistics<N extends num> extends DataEntry {
  double length;

  N min;

  N max;

  N center;

  num sum;

  num squaresSum;
  double mean;

  double standardDeviation;

  DataStatistics? lowerStatistics;

  DataStatistics? upperStatistics;

  DataStatistics(
    num length,
    this.min,
    this.max,
    this.center, {
    double? mean,
    double? standardDeviation,
    num? sum,
    num? squaresSum,
    this.lowerStatistics,
    this.upperStatistics,
  })  : length = length.toDouble(),
        sum = sum ?? (mean! * length),
        squaresSum =
            squaresSum ?? ((standardDeviation! * standardDeviation) * length),
        mean = mean ?? (sum! / length),
        standardDeviation =
            standardDeviation ?? math.sqrt(squaresSum! / length);

  factory DataStatistics._empty(List<N> list) {
    var zero = list.castElement(0);
    return DataStatistics(0, zero, zero, zero, sum: zero, squaresSum: zero);
  }

  factory DataStatistics._single(N n) {
    return DataStatistics(1, n, n, n,
        sum: n, squaresSum: n * n, mean: n.toDouble(), standardDeviation: 0);
  }

  factory DataStatistics.compute(List<N> list,
      {bool computeLowerAndUpper = true, bool keepSeries = false}) {
    var length = list.length;
    if (length == 0) return DataStatistics._empty(list);
    if (length == 1) return DataStatistics._single(list.first);

    var listSorted = List<N>.from(list);
    listSorted.sort();

    var first = listSorted.first;
    var min = first;
    var max = listSorted.last;
    var centerIndex = listSorted.length ~/ 2;
    var center = listSorted[centerIndex];

    num sum = first;
    var squaresSum = first * first;

    for (var i = 1; i < length; ++i) {
      var n = listSorted[i];
      sum += n;
      squaresSum += n * n;
    }

    var mean = sum / length;

    var standardDeviation = math.sqrt(squaresSum / length);

    DataStatistics? lowerStatistics;
    DataStatistics? upperStatistics;

    if (computeLowerAndUpper) {
      var lower = listSorted.sublist(0, centerIndex);
      var upper = listSorted.sublist(centerIndex);

      lowerStatistics = DataStatistics.compute(lower,
          computeLowerAndUpper: false, keepSeries: false);
      upperStatistics = DataStatistics.compute(upper,
          computeLowerAndUpper: false, keepSeries: false);
    }

    var statistics = DataStatistics(
      length,
      min,
      max,
      center,
      sum: sum,
      squaresSum: squaresSum,
      mean: mean,
      standardDeviation: standardDeviation,
      lowerStatistics: lowerStatistics,
      upperStatistics: upperStatistics,
    );

    if (keepSeries) {
      statistics.series = list;
    }

    return statistics;
  }

  List<N>? series;

  bool isMeanInRange(double minMean, double maxMean,
      [double minDeviation = double.negativeInfinity,
      double maxDeviation = double.infinity]) {
    return (mean >= minMean && mean <= maxMean) &&
        (standardDeviation >= minDeviation &&
            standardDeviation <= maxDeviation);
  }

  double get squaresMean => squaresSum / length;

  @override
  String toString({int precision = 4}) {
    if (length == 0) {
      return '{empty}';
    }

    var minStr = precision > 0 ? formatDecimal(min, precision: precision) : min;
    var maxStr = precision > 0 ? formatDecimal(max, precision: precision) : max;
    var centerStr =
        precision > 0 ? formatDecimal(center, precision: precision) : center;

    //var sumStr = precision > 0 ? formatDecimal(sum, precision: precision) : sum;
    //var squaresSumStr = precision > 0 ? formatDecimal(squaresSum, precision: precision) : squaresSum;

    var meanStr =
        precision > 0 ? formatDecimal(mean, precision: precision) : mean;
    var standardDeviationStr = precision > 0
        ? formatDecimal(standardDeviation, precision: precision)
        : standardDeviation;

    //return '{ min..max: $minStr .. $maxStr ; mean: $meanStr (+-$standardDeviationStr) ; #$length }';

    return '{~$meanStr +-$standardDeviationStr [$minStr..($centerStr)..$maxStr] #$length}';
  }

  DataStatistics<double> operator /(DataStatistics other) {
    return DataStatistics(
      length / other.length,
      min / other.min,
      max / other.max,
      center / other.center,
      sum: sum / other.sum,
      squaresSum: squaresSum / other.squaresSum,
      mean: mean / other.mean,
      standardDeviation: standardDeviation / other.standardDeviation,
    );
  }

  DataStatistics<double> operator +(DataStatistics other) {
    return DataStatistics(
      length + other.length,
      math.min(min.toDouble(), other.min.toDouble()),
      math.max(max.toDouble(), other.max.toDouble()),
      (center + other.center) / 2,
      sum: sum + other.sum,
      squaresSum: squaresSum + other.squaresSum,
      mean: (sum + other.sum) / (length + other.length),
      standardDeviation:
          math.sqrt((squaresSum + other.squaresSum) / (length + other.length)),
    );
  }

  @override
  List<String> getDataFields() =>
      ['mean', 'standardDeviation', 'length', 'min', 'max'];

  @override
  List getDataValues() => [mean, standardDeviation, length, min, max];
}

abstract class DataEntry {
  List<String> getDataFields();

  List getDataValues();

  Map<String, dynamic> getDataMap() =>
      Map.fromIterables(getDataFields(), getDataValues());
}

extension DataEntryExtension<E extends DataEntry> on List<E> {
  String generateCSV({String separator = ',', List<String>? fieldsNames}) {
    if (isEmpty) return '';

    var csv = StringBuffer();

    fieldsNames ??= first.getDataFields();

    {
      var head = fieldsNames.map(_normalizeLine).join(separator);
      csv.write(head);
      csv.write('\n');
    }

    for (var e in this) {
      var values = e.getDataValues();
      var line = values.map((e) => _normalizeLine('$e')).join(separator);
      csv.write(line);
      csv.write('\n');
    }

    return csv.toString();
  }
}

final _REGEXP_NEW_LINE = RegExp(r'[\r\n]');

String _normalizeLine(String e) => e.replaceAll(_REGEXP_NEW_LINE, ' ');

extension SeriesMapExtension<N extends num> on Map<String, List<N>?> {
  N _toN(num n) => (N == int ? n.toInt() : n.toDouble()) as N;

  String csvFileName(String prefix, String name) {
    var now = DateTime.now().millisecondsSinceEpoch;
    var csvFileName = '$prefix--$name--$now.csv';
    return csvFileName;
  }

  String generateCSV(
      {String separator = ',', N? nullValue, int firstEntryIndex = 1}) {
    if (isEmpty) return '';

    var csv = StringBuffer();

    var keys = this.keys;

    {
      var head = keys.map(_normalizeLine).join(separator);
      csv.write('#$separator');
      csv.write(head);
      csv.write('\n');
    }

    nullValue ??= _toN(0);

    var totalLines = values.map((e) => e?.length ?? 0).toList().statistics.max;

    for (var i = 0; i < totalLines; ++i) {
      var line = StringBuffer();
      line.write('${i + firstEntryIndex}');

      for (var k in keys) {
        var e = this[k]?.getValueIfExists(i) ?? nullValue;
        line.write(separator);
        line.write(e);
      }

      csv.write(line);
      csv.write('\n');
    }

    return csv.toString();
  }
}
