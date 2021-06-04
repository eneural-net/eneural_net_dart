import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:eneural_net/eneural_net.dart';
import 'package:swiss_knife/swiss_knife.dart';

extension Int32x4Extension on Int32x4 {
  /// Converts to a [Float32x4].
  Float32x4 toFloat32x4() =>
      Float32x4(x.toDouble(), y.toDouble(), z.toDouble(), w.toDouble());

  /// Filter this with [mapper].
  Int32x4 filter(Int32x4 Function(Int32x4 e) mapper) => mapper(this);

  /// Filter each value with [mapper] and return a [Int32x4].
  Int32x4 filterValues(int Function(int e) mapper) {
    return Int32x4(
      mapper(x),
      mapper(y),
      mapper(z),
      mapper(w),
    );
  }

  /// Filter each value with [mapper] and return a [Float32x4].
  Float32x4 filterToDoubleValues(double Function(int e) mapper) {
    return Float32x4(
      mapper(x),
      mapper(y),
      mapper(z),
      mapper(w),
    );
  }

  /// Map using [mapper].
  T map<T>(T Function(Int32x4 e) mapper) => mapper(this);

  /// Returns values as `List<int>`.
  List<int> toInts() => <int>[x, y, z, w];

  Int32x4 operator *(Int32x4 other) => Int32x4(
        x * other.x,
        y * other.y,
        z * other.z,
        w * other.w,
      );

  Int32x4 operator ~/(Int32x4 other) => Int32x4(
        x ~/ other.x,
        y ~/ other.y,
        z ~/ other.z,
        w ~/ other.w,
      );

  /// Returns the minimal lane value.
  int get minInLane {
    var min = x;
    if (y < min) min = y;
    if (z < min) min = z;
    if (w < min) min = w;
    return min;
  }

  /// Returns the maximum lane value.
  int get maxInLane {
    var max = x;
    if (y > max) max = y;
    if (z > max) max = z;
    if (w > max) max = w;
    return max;
  }

  /// Sum lane.
  int get sumLane => x + y + z + w;

  /// Sum part of the lane, until [size].
  int sumLanePartial(int size) {
    switch (size) {
      case 1:
        return x;
      case 2:
        return x + y;
      case 3:
        return x + y + z;
      case 4:
        return x + y + z + w;
      default:
        throw StateError('Invalid length: $size / 4');
    }
  }

  /// Sum lane squares.
  int get sumSquaresLane => (x * x) + (y * y) + (z * z) + (w * w);

  /// Sum part of the lane squares, until [size].
  int sumSquaresLanePartial(int size) {
    switch (size) {
      case 1:
        return (x * x);
      case 2:
        return (x * x) + (y * y);
      case 3:
        return (x * x) + (y * y) + (z * z);
      case 4:
        return (x * x) + (y * y) + (z * z) + (w * w);
      default:
        throw StateError('Invalid length: $size / 4');
    }
  }

  /// Returns true if equals to [other] values.
  bool equalsValues(Int32x4 other) {
    var diff = this - other;
    return diff.x == 0 && diff.y == 0 && diff.z == 0 && diff.w == 0;
  }
}

extension Float32x4Extension on Float32x4 {
  /// Converts to a [Int32x4].
  Int32x4 toInt32x4() => Int32x4(x.toInt(), y.toInt(), z.toInt(), w.toInt());

  /// Perform a `toInt()` in each value and return a [Float32x4].
  Float32x4 toIntAsFloat32x4() => Float32x4(x.toInt().toDouble(),
      y.toInt().toDouble(), z.toInt().toDouble(), w.toInt().toDouble());

  /// Filter this with [mapper].
  Float32x4 filter(Float32x4 Function(Float32x4 e) filter) => filter(this);

  /// Filter each value with [mapper] and return a [Float32x4].
  Float32x4 filterValues(double Function(double e) mapper) {
    return Float32x4(
      mapper(x),
      mapper(y),
      mapper(z),
      mapper(w),
    );
  }

  /// Filter each value with [mapper] and return a [Int32x4].
  Int32x4 filterToIntValues(int Function(double e) mapper) {
    return Int32x4(
      mapper(x),
      mapper(y),
      mapper(z),
      mapper(w),
    );
  }

  /// Map using [mapper].
  T map<T>(T Function(Float32x4 e) mapper) => mapper(this);

  /// Returns values as `List<double>`.
  List<double> toDoubles() => <double>[x, y, z, w];

  /// Returns the minimum lane value.
  double get minInLane {
    var min = x;
    if (y < min) min = y;
    if (z < min) min = z;
    if (w < min) min = w;
    return min;
  }

  /// Returns the maximum lane value.
  double get maxInLane {
    var max = x;
    if (y > max) max = y;
    if (z > max) max = z;
    if (w > max) max = w;
    return max;
  }

  /// Sum lane.
  double get sumLane => x + y + z + w;

  /// Sum part of the lane, until [size].
  double sumLanePartial(int size) {
    switch (size) {
      case 1:
        return x;
      case 2:
        return x + y;
      case 3:
        return x + y + z;
      case 4:
        return x + y + z + w;
      default:
        throw StateError('Invalid length: $size / 4');
    }
  }

  /// Sum lane squares.
  double get sumSquaresLane => (x * x) + (y * y) + (z * z) + (w * w);

  /// Sum part of the lane squares, until [size].
  double sumSquaresLanePartial(int size) {
    switch (size) {
      case 1:
        return (x * x);
      case 2:
        return (x * x) + (y * y);
      case 3:
        return (x * x) + (y * y) + (z * z);
      case 4:
        return (x * x) + (y * y) + (z * z) + (w * w);
      default:
        throw StateError('Invalid length: $size / 4');
    }
  }

  /// Returns true if equals to [other] values.
  bool equalsValues(Float32x4 other) {
    var diff = this - other;
    return diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0 && diff.w == 0.0;
  }
}

extension ListExtension<T> on List<T> {
  int get lastIndex => length - 1;

  T getReversed(int reversedIndex) => this[lastIndex - reversedIndex];

  T? getValueIfExists(int index) =>
      index >= 0 && index < length ? this[index] : null;

  int setAllWithValue(T n) {
    var lng = length;
    for (var i = 0; i < lng; ++i) {
      this[i] = n;
    }
    return lng;
  }

  int setAllWith(T Function(int index, T value) f) {
    var lng = length;
    for (var i = 0; i < lng; ++i) {
      this[i] = f(i, this[i]);
    }
    return lng;
  }

  int setAllWithList(List<T> list, [int offset = 0]) {
    var lng = length;
    for (var i = 0; i < lng; ++i) {
      this[i] = list[offset + i];
    }
    return lng;
  }

  bool allEquals(T element) {
    if (length == 0) return false;

    for (var e in this) {
      if (e != element) return false;
    }

    return true;
  }

  List<String> toStringElements() => map((e) => '$e').toList();

  int computeHashcode() {
    return ListEquality<T>().hash(this);
  }

  int ensureMaximumSize(int maximumSize,
      {bool removeFromEnd = false, int removeExtras = 0}) {
    var toRemove = length - maximumSize;
    if (toRemove <= 0) return 0;

    if (removeExtras > 0) {
      toRemove += removeExtras;
    }

    if (removeFromEnd) {
      return this.removeFromEnd(toRemove);
    } else {
      return removeFromBegin(toRemove);
    }
  }

  int removeFromBegin(int amount) {
    if (amount <= 0) return 0;
    var length = this.length;
    if (amount > length) amount = length;
    removeRange(0, amount);
    return amount;
  }

  int removeFromEnd(int amount) {
    if (amount <= 0) return 0;
    var length = this.length;
    if (amount > length) amount = length;
    removeRange(length - amount, length);
    return amount;
  }

  List<double> asDoubles() => this is List<double>
      ? this as List<double>
      : map((v) => parseDouble(v)!).toList();

  List<int> asInts() =>
      this is List<int> ? this as List<int> : map((v) => parseInt(v)!).toList();
}

extension SetExtension<T> on Set<T> {
  bool allEquals(T element) {
    if (length == 0) return false;

    for (var e in this) {
      if (e != element) return false;
    }

    return true;
  }

  List<String> toStringElements() => map((e) => '$e').toList();

  int computeHashcode() {
    return SetEquality<T>().hash(this);
  }
}

extension IterableExtension<T> on Iterable<T> {
  Map<G, List<T>> groupBy<G>(G Function(T e) grouper) {
    var groups = <G, List<T>>{};

    for (var e in this) {
      var g = grouper(e);
      var list = groups.putIfAbsent(g, () => <T>[]);
      list.add(e);
    }

    return groups;
  }
}

extension ListNumExtension<N extends num> on List<N> {
  N castElement(num n) {
    if (N == int) {
      return n.toInt() as N;
    } else {
      return n.toDouble() as N;
    }
  }

  List<T> mapToList<T>(T Function(N n) f) => map(f).toList();

  Set<T> mapToSet<T>(T Function(N n) f) => map(f).toSet();

  List<int> toInts() => mapToList((e) => e.toInt());

  List<double> toDoubles() => mapToList((e) => e.toDouble());

  List<String> toStrings() => mapToList((e) => e.toString());

  DataStatistics<num> get statistics => DataStatistics.compute(this);

  N get sum {
    var length = this.length;
    if (length == 0) return castElement(0);

    num total = first;

    for (var i = 1; i < length; ++i) {
      total += this[i];
    }

    return castElement(total);
  }

  N get sumSquares {
    var length = this.length;
    if (length == 0) return castElement(0);

    var first = this.first;

    var total = first * first;

    for (var i = 1; i < length; ++i) {
      var n = this[i];
      total += n * n;
    }

    return castElement(total);
  }

  double get mean {
    return sum / length;
  }

  double get standardDeviation {
    var length = this.length;
    if (length == 0) return 0;

    var average = mean;

    var total = 0.0;

    for (var i = 0; i < length; ++i) {
      var n = this[i] - average;
      total += n * n;
    }

    var deviation = sqrt(total / length);

    return deviation;
  }

  double get squaresMean => sumSquares / length;

  List<N> get square => map((n) => castElement(n * n)).toList();

  List<N> get abs => map((n) => castElement(n.abs())).toList();

  List<double> movingAverage(int samplesSize) {
    var length = this.length;
    if (samplesSize >= length) return <double>[mean];

    var movingAverage = <double>[];
    for (var i = 0; i < length; ++i) {
      var end = i + samplesSize;
      if (end > length) break;

      var total = 0.0;
      for (var j = i; j < end; ++j) {
        var e = this[j];
        total += e;
      }

      var average = total / samplesSize;
      movingAverage.add(average);
    }

    return movingAverage;
  }

  List<double> mergeBlocks(int blocksSize) {
    var length = this.length;
    if (length <= blocksSize) return [mean];

    var merge = <double>[];

    for (var i = 0; i < length; i += blocksSize) {
      var end = i + blocksSize;
      if (end > length) end = length;
      var block = sublist(i, end);
      merge.add(block.mean);
    }

    return merge;
  }

  List<num> diff(List<N> other) =>
      List<num>.generate(length, (i) => this[i] - other[i]);

  List<num> diffFromSignal<E, T extends Signal<N, E, T>>(T signal) =>
      List<num>.generate(length, (i) => this[i] - signal.getValue(i));

  List<num> operator -(List<num> other) {
    return List.generate(length, (i) => this[i] - other[i]);
  }

  List<num> operator +(List<num> other) {
    return List.generate(length, (i) => this[i] + other[i]);
  }

  List<num> operator *(List<num> other) {
    return List.generate(length, (i) => this[i] * other[i]);
  }

  List<double> operator /(List<num> other) {
    return List.generate(length, (i) => this[i] / other[i]);
  }

  List<int> operator ~/(List<num> other) {
    return List.generate(length, (i) => this[i] ~/ other[i]);
  }
}

extension ListDoubleExtension on List<double> {
  double castElement(num n) {
    return n.toDouble();
  }

  List<double> toDoubles() => toList();

  DataStatistics<num> get statistics => DataStatistics.compute(this);

  DataStatistics<num> get statisticsWithSeries =>
      DataStatistics.compute(this, keepSeries: true);

  double get sum {
    var length = this.length;
    if (length == 0) return 0;

    var total = first;

    for (var i = 1; i < length; ++i) {
      total += this[i];
    }

    return total;
  }

  double get sumSquares {
    var length = this.length;
    if (length == 0) return 0.0;

    var first = this.first;

    var total = first * first;

    for (var i = 1; i < length; ++i) {
      var n = this[i];
      total += n * n;
    }

    return total;
  }

  double get mean {
    return sum / length;
  }

  double get standardDeviation {
    var length = this.length;
    if (length == 0) return 0.0;

    var average = mean;

    var total = 0.0;

    for (var i = 0; i < length; ++i) {
      var n = this[i] - average;
      total += n * n;
    }

    var deviation = sqrt(total / length);

    return deviation;
  }

  List<double> get square => map((n) => n * n).toList();

  double get squaresMean => sumSquares / length;

  List<double> get abs => map((n) => n.abs()).toList();

  List<double> diff(List<double> other) =>
      List<double>.generate(length, (i) => this[i] - other[i]);

  List<double> diffFromSignal<E, T extends Signal<num, E, T>>(T signal) =>
      List<double>.generate(length, (i) => this[i] - signal.getValue(i));

  List<double> operator -(List<double> other) {
    return List.generate(length, (i) => this[i] - other[i]);
  }

  List<double> operator +(List<double> other) {
    return List.generate(length, (i) => this[i] + other[i]);
  }

  List<double> operator *(List<double> other) {
    return List.generate(length, (i) => this[i] * other[i]);
  }

  List<double> operator /(List<double> other) {
    return List.generate(length, (i) => this[i] / other[i]);
  }

  List<int> operator ~/(List<double> other) {
    return List.generate(length, (i) => this[i] ~/ other[i]);
  }
}

extension ListIntExtension on List<int> {
  int castElement(num n) {
    return n.toInt();
  }

  List<int> toInts() => toList();

  DataStatistics<num> get statistics => DataStatistics.compute(this);

  DataStatistics<num> get statisticsWithSeries =>
      DataStatistics.compute(this, keepSeries: true);

  int get sum {
    var length = this.length;
    if (length == 0) return 0;

    var total = first;

    for (var i = 1; i < length; ++i) {
      total += this[i];
    }

    return total;
  }

  int get sumSquares {
    var length = this.length;
    if (length == 0) return 0;

    var first = this.first;

    var total = first * first;

    for (var i = 1; i < length; ++i) {
      var n = this[i];
      total += n * n;
    }

    return total;
  }

  double get mean {
    return sum / length;
  }

  double get standardDeviation {
    var length = this.length;
    if (length == 0) return 0;

    var average = mean;

    var total = 0.0;

    for (var i = 0; i < length; ++i) {
      var n = this[i] - average;
      total += n * n;
    }

    var deviation = sqrt(total / length);

    return deviation;
  }

  List<int> get square => map((n) => n * n).toList();

  List<int> get abs => map((n) => n.abs()).toList();
}

extension NumExtension on num {
  num get square => this * this;

  double get squareRoot => sqrt(this);

  double get naturalExponent => exp(this);

  num clamp(num min, num max) {
    if (this < min) {
      return min;
    } else if (this > max) {
      return max;
    } else {
      return this;
    }
  }

  int signWithZeroTolerance([double zeroTolerance = 1.0E-20]) {
    if (this > 0) {
      return this < zeroTolerance ? 0 : 1;
    } else {
      return this > -zeroTolerance ? 0 : -1;
    }
  }
}

extension DoubleExtension on double {
  double get square => this * this;

  double clamp(double min, double max) {
    if (this < min) {
      return min;
    } else if (this > max) {
      return max;
    } else {
      return this;
    }
  }

  int signWithZeroTolerance([double zeroTolerance = 0.0000000000001]) {
    if (this > 0) {
      return this < zeroTolerance ? 0 : 1;
    } else {
      return this > -zeroTolerance ? 0 : -1;
    }
  }
}

extension IntExtension on int {
  int get square => this * this;

  int clamp(int min, int max) {
    if (this < min) {
      return min;
    } else if (this > max) {
      return max;
    } else {
      return this;
    }
  }
}

class Int32x4Equality implements Equality<Int32x4> {
  @override
  bool equals(Int32x4 e1, Int32x4 e2) => e1.equalsValues(e2);

  @override
  int hash(Int32x4 e) =>
      e.x.hashCode ^ e.y.hashCode ^ e.z.hashCode ^ e.w.hashCode;

  @override
  bool isValidKey(Object? o) {
    return o is Int32x4;
  }
}

class Float32x4Equality implements Equality<Float32x4> {
  @override
  bool equals(Float32x4 e1, Float32x4 e2) => e1.equalsValues(e2);

  @override
  int hash(Float32x4 e) =>
      e.x.hashCode ^ e.y.hashCode ^ e.z.hashCode ^ e.w.hashCode;

  @override
  bool isValidKey(Object? o) {
    return o is Float32x4;
  }
}

extension DurationExtension on Duration {
  String toStringUnit({
    bool days = true,
    bool hours = true,
    bool minutes = true,
    bool seconds = true,
    bool milliseconds = true,
    bool microseconds = true,
  }) {
    if (days && inDays > 0) {
      return '$inDays d';
    } else if (hours && inHours > 0) {
      return '$inHours h';
    } else if (minutes && inMinutes > 0) {
      return '$inMinutes min';
    } else if (seconds && inSeconds > 0) {
      return '$inSeconds sec';
    } else if (milliseconds && inMilliseconds > 0) {
      return '$inMilliseconds ms';
    } else if (microseconds && inMicroseconds > 0) {
      return '$inMicroseconds μs';
    } else {
      return toString();
    }
  }
}
