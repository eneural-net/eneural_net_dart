import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:eneural_net/eneural_net.dart';

extension Int32x4Extension on Int32x4 {
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

  int get sumLane => x + y + z + w;

  bool equalsValues(Int32x4 other) {
    var diff = this - other;
    return diff.x == 0 && diff.y == 0 && diff.z == 0 && diff.w == 0;
  }
}

extension Float32x4Extension on Float32x4 {
  double get sumLane => x + y + z + w;

  bool equalsValues(Float32x4 other) {
    var diff = this - other;
    return diff.x == 0.0 && diff.y == 0.0 && diff.z == 0.0 && diff.w == 0.0;
  }
}

extension ListExtension<T> on List<T> {
  void setAllWithValue(T n) {
    for (var i = 0; i < length; ++i) {
      this[i] = n;
    }
  }

  void setAllWith(T Function(int index, T value) f) {
    for (var i = 0; i < length; ++i) {
      this[i] = f(i, this[i]);
    }
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

    num total = first;

    for (var i = 1; i < length; ++i) {
      var n = this[i];
      total += n * n;
    }

    return castElement(total);
  }

  double get mean {
    return sum / length;
  }

  List<N> get square => map((n) => castElement(n * n)).toList();

  List<N> get abs => map((n) => castElement(n.abs())).toList();

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

  double get sum {
    var length = this.length;
    if (length == 0) return 0;

    var total = first;

    for (var i = 1; i < length; ++i) {
      total += this[i];
    }

    return total;
  }

  double get mean {
    return sum / length;
  }

  List<double> get square => map((n) => n * n).toList();

  List<double> get abs => map((n) => n.abs()).toList();
}

extension ListIntExtension on List<int> {
  int castElement(num n) {
    return n.toInt();
  }

  int get sum {
    var length = this.length;
    if (length == 0) return 0;

    var total = first;

    for (var i = 1; i < length; ++i) {
      total += this[i];
    }

    return total;
  }

  double get mean {
    return sum / length;
  }

  List<int> get square => map((n) => n * n).toList();

  List<int> get abs => map((n) => n.abs()).toList();
}

extension NumExtension on num {
  num get square => this * this;

  double get squareRoot => sqrt(this);

  double get naturalExponent => exp(this);
}

extension DoubleExtension on double {
  double get square => this * this;
}

extension IntExtension on int {
  int get square => this * this;
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
