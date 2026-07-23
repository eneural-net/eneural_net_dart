import 'dart:math' as dart_math;

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_fast_math.dart' as fast_math;
import 'package:test/test.dart';

/// A plain (non-SIMD) [Signal] implementation used to exercise the abstract
/// contract of the base class for entry block sizes other than 4.
///
/// An entry is a `List<double>` of [entryBlockSize] values.
class ListSignal extends Signal<double, List<double>, ListSignal> {
  @override
  final int entryBlockSize;

  /// When true the `createEntryN`/`createEntryFrom` factories delegate to the
  /// base class, which rejects them as unsupported.
  final bool useBaseEntryFactories;

  final List<List<double>> _entries;

  final int _size;

  ListSignal(
    this.entryBlockSize,
    int size, {
    this.useBaseEntryFactories = false,
  }) : _size = size,
       _entries = List.generate(
         Signal.calcNeededBlocks(size, entryBlockSize),
         (_) => List<double>.filled(entryBlockSize, 0.0),
       );

  ListSignal._(
    this.entryBlockSize,
    this._entries,
    this._size,
    this.useBaseEntryFactories,
  );

  @override
  String get format => 'List$entryBlockSize';

  @override
  int calcEntriesCapacityForSize(int size) =>
      Signal.calcNeededBlocks(size, entryBlockSize);

  @override
  double get zero => 0.0;

  @override
  double get one => 1.0;

  @override
  double toN(num n) => n.toDouble();

  @override
  String nToString(double n) => '$n';

  @override
  int get length => _size;

  @override
  int get entriesLength => _entries.length;

  @override
  int get capacity => _entries.length * entryBlockSize;

  @override
  List<List<double>> get entries => _entries.toList();

  @override
  List<double> getEntry(int index) => _entries[index];

  @override
  void setEntry(int index, List<double> entry) =>
      _entries[index] = List<double>.from(entry);

  @override
  void addToEntry(int index, List<double> entry) {
    var current = _entries[index];
    for (var i = 0; i < entryBlockSize; ++i) {
      current[i] += entry[i];
    }
  }

  @override
  void subtractToEntry(int index, List<double> entry) {
    var current = _entries[index];
    for (var i = 0; i < entryBlockSize; ++i) {
      current[i] -= entry[i];
    }
  }

  @override
  List<double> getEntryFilteredX4(
    int index,
    List<double> Function(List<double> entry) filter,
  ) => filter(_entries[index]);

  @override
  List<double> getEntryFiltered(int index, double Function(double n) filter) =>
      _entries[index].map(filter).toList();

  @override
  double getValueFromEntry(List<double> entry, int offset) => entry[offset];

  @override
  List<double> setValueFromEntry(List<double> entry, int offset, double value) {
    var copy = List<double>.from(entry);
    copy[offset] = value;
    return copy;
  }

  @override
  List<double> addValueFromEntry(List<double> entry, int offset, double value) {
    var copy = List<double>.from(entry);
    copy[offset] += value;
    return copy;
  }

  // The `createEntryN` below delegate to `super` when N is bigger than
  // [entryBlockSize], so the base class rejects the unsupported operation.

  @override
  List<double> createEntry1(double v0) {
    if (useBaseEntryFactories || entryBlockSize < 1) {
      return super.createEntry1(v0);
    }
    return _pad([v0]);
  }

  @override
  List<double> createEntry2(double v0, double v1) {
    if (useBaseEntryFactories || entryBlockSize < 2) {
      return super.createEntry2(v0, v1);
    }
    return _pad([v0, v1]);
  }

  @override
  List<double> createEntry3(double v0, double v1, double v2) {
    if (useBaseEntryFactories || entryBlockSize < 3) {
      return super.createEntry3(v0, v1, v2);
    }
    return _pad([v0, v1, v2]);
  }

  @override
  List<double> createEntry4(double v0, double v1, double v2, double v3) {
    if (useBaseEntryFactories || entryBlockSize < 4) {
      return super.createEntry4(v0, v1, v2, v3);
    }
    return _pad([v0, v1, v2, v3]);
  }

  @override
  List<double> createEntryFrom(
    List<double> other, [
    double? v0,
    double? v1,
    double? v2,
    double? v3,
  ]) {
    if (useBaseEntryFactories) {
      return super.createEntryFrom(other, v0, v1, v2, v3);
    }
    var values = [v0, v1, v2, v3];
    return List<double>.generate(entryBlockSize, (i) => values[i] ?? other[i]);
  }

  /// Completes an entry with zeroes up to [entryBlockSize].
  List<double> _pad(List<double> values) {
    while (values.length < entryBlockSize) {
      values.add(0.0);
    }
    return values;
  }

  @override
  List<double> createEntryFullOf(double v) =>
      List<double>.filled(entryBlockSize, v);

  @override
  List<double> get entryEmpty => List<double>.filled(entryBlockSize, 0.0);

  static final dart_math.Random _random = dart_math.Random();

  @override
  double createRandomValue(double scale, [dart_math.Random? rand]) {
    rand ??= _random;
    return (rand.nextDouble() * (scale * 2)) - scale;
  }

  @override
  List<double> createRandomEntry(double scale, [dart_math.Random? rand]) =>
      List<double>.generate(
        entryBlockSize,
        (_) => createRandomValue(scale, rand),
      );

  @override
  ListSignal copy() => ListSignal._(
    entryBlockSize,
    _entries.map((e) => List<double>.from(e)).toList(),
    _size,
    useBaseEntryFactories,
  );

  @override
  ListSignal createInstance(int size) => ListSignal(entryBlockSize, size);

  @override
  ListSignal createRandomInstance(
    int size,
    double randomScale, [
    dart_math.Random? rand,
  ]) {
    var instance = createInstance(size);
    for (var i = 0; i < instance.entriesLength; ++i) {
      instance.setEntry(i, createRandomEntry(randomScale, rand));
    }
    return instance;
  }

  @override
  ListSignal createInstanceWithEntries(int size, List<List<double>> entries) {
    ensureEntriesLengthMod(entries);
    return ListSignal._(
      entryBlockSize,
      entries.map((e) => List<double>.from(e)).toList(),
      size,
      useBaseEntryFactories,
    );
  }

  @override
  void ensureEntriesLengthMod(List<List<double>> entries) {}

  @override
  List<double> entryOperationSum(List<double> e1, List<double> e2) =>
      List<double>.generate(entryBlockSize, (i) => e1[i] + e2[i]);

  @override
  List<double> entryOperationSubtract(List<double> e1, List<double> e2) =>
      List<double>.generate(entryBlockSize, (i) => e1[i] - e2[i]);

  @override
  List<double> entryOperationMultiply(List<double> e1, List<double> e2) =>
      List<double>.generate(entryBlockSize, (i) => e1[i] * e2[i]);

  @override
  List<double> entryOperationDivide(List<double> e1, List<double> e2) =>
      List<double>.generate(entryBlockSize, (i) => e1[i] / e2[i]);

  @override
  double entryOperationSumLane(List<double> entry) =>
      entry.fold<double>(0.0, (t, v) => t + v);

  @override
  double entryOperationSumLanePartial(List<double> entry, int size) {
    var total = 0.0;
    for (var i = 0; i < size; ++i) {
      total += entry[i];
    }
    return total;
  }

  @override
  double entryOperationSumSquaresLane(List<double> entry) =>
      entry.fold<double>(0.0, (t, v) => t + (v * v));

  @override
  double entryOperationSumSquaresLanePartial(List<double> entry, int size) {
    var total = 0.0;
    for (var i = 0; i < size; ++i) {
      total += entry[i] * entry[i];
    }
    return total;
  }

  @override
  void multiplyTo(ListSignal other, ListSignal destiny) {
    for (var i = 0; i < entriesLength; ++i) {
      destiny.setEntry(
        i,
        entryOperationMultiply(getEntry(i), other.getEntry(i)),
      );
    }
  }

  @override
  ListSignal multiply(ListSignal other) {
    var destiny = createInstance(length);
    multiplyTo(other, destiny);
    return destiny;
  }

  @override
  void subtractTo(ListSignal other, ListSignal destiny) {
    for (var i = 0; i < entriesLength; ++i) {
      destiny.setEntry(
        i,
        entryOperationSubtract(getEntry(i), other.getEntry(i)),
      );
    }
  }

  @override
  ListSignal subtract(ListSignal other) {
    var destiny = createInstance(length);
    subtractTo(other, destiny);
    return destiny;
  }

  @override
  void multiplyAllEntriesTo(List<double> entry, ListSignal destiny) {
    for (var i = 0; i < entriesLength; ++i) {
      destiny.setEntry(i, entryOperationMultiply(getEntry(i), entry));
    }
  }

  @override
  void subtractAllEntriesTo(List<double> entry, ListSignal destiny) {
    for (var i = 0; i < entriesLength; ++i) {
      destiny.setEntry(i, entryOperationSubtract(getEntry(i), entry));
    }
  }

  @override
  void multiplyAllEntriesAddingTo(List<double> entry, ListSignal destiny) {
    for (var i = 0; i < entriesLength; ++i) {
      destiny.setEntry(
        i,
        entryOperationSum(
          destiny.getEntry(i),
          entryOperationMultiply(getEntry(i), entry),
        ),
      );
    }
  }

  @override
  ListSignal multiplyEntries(List<double> entry) {
    var destiny = createInstance(length);
    multiplyAllEntriesTo(entry, destiny);
    return destiny;
  }

  @override
  List<double> normalizeEntry(List<double> entry, Scale<double> scale) =>
      entry.map(scale.normalize).toList();
}

void main() {
  group('Signal contract: entry block sizes 1..4', () {
    for (var blockSize in [1, 2, 3, 4]) {
      group('entryBlockSize $blockSize', () {
        test('valuesToEntries packs the values', () {
          var prototype = ListSignal(blockSize, 0);

          for (var size = 0; size <= 3 * blockSize + 2; ++size) {
            var values = List<double>.generate(size, (i) => (i + 1).toDouble());

            var entries = prototype.valuesToEntries(values);

            expect(
              entries.length,
              equals(Signal.calcNeededBlocks(size, blockSize)),
              reason: 'block $blockSize, size $size',
            );

            // Flattening the entries must give back the values plus padding:
            var flat = entries.expand((e) => e).toList();
            expect(
              flat.sublist(0, size),
              equals(values),
              reason: 'block $blockSize, size $size',
            );
          }
        });

        test('createInstanceWithValues round-trips the values', () {
          var prototype = ListSignal(blockSize, 0);

          for (var size = 1; size <= 3 * blockSize + 2; ++size) {
            var values = List<double>.generate(size, (i) => (i + 1).toDouble());

            var signal = prototype.createInstanceWithValues(values);

            expect(signal.length, equals(size));
            expect(signal.values, equals(values));
            expect(signal.entryBlockSize, equals(blockSize));
          }
        });

        test('lastEntryLength and valuesEntriesLength are consistent', () {
          for (var size = 0; size <= 3 * blockSize + 2; ++size) {
            var signal = ListSignal(blockSize, size);

            expect(
              signal.valuesEntriesLength,
              equals(Signal.calcNeededBlocks(size, blockSize)),
            );

            if (size == 0) {
              expect(signal.lastEntryLength, equals(0));
            } else {
              expect(signal.lastEntryLength >= 1, isTrue);
              expect(signal.lastEntryLength <= blockSize, isTrue);
              expect(
                (signal.valuesEntriesLength - 1) * blockSize +
                    signal.lastEntryLength,
                equals(size),
                reason: 'block $blockSize, size $size',
              );
            }
          }
        });

        test('setExtraValues fills the padding without touching values', () {
          for (var size = 1; size <= 3 * blockSize + 2; ++size) {
            var values = List<double>.generate(size, (i) => (i + 1).toDouble());

            var signal = ListSignal(
              blockSize,
              0,
            ).createInstanceWithValues(values);

            signal.setExtraValues(-1.0);

            expect(
              signal.values,
              equals(values),
              reason: 'block $blockSize, size $size',
            );

            for (var i = size; i < signal.capacity; ++i) {
              expect(
                signal.getValue(i),
                equals(-1.0),
                reason: 'padding $i of block $blockSize, size $size',
              );
            }
          }
        });

        test('computeSumSquares ignores the padding', () {
          for (var size = 1; size <= 2 * blockSize + 1; ++size) {
            var values = List<double>.generate(size, (i) => (i + 1).toDouble());
            var expected = values.fold<double>(0, (t, v) => t + (v * v));

            var signal = ListSignal(
              blockSize,
              0,
            ).createInstanceWithValues(values);
            signal.setExtraValues(100.0);

            expect(
              signal.computeSumSquares(),
              closeTo(expected, 1e-9),
              reason: 'block $blockSize, size $size',
            );
          }
        });

        test('the generic values getter reads every value', () {
          var values = List<double>.generate(
            3 * blockSize + 1,
            (i) => (i + 1).toDouble(),
          );

          var signal = ListSignal(
            blockSize,
            0,
          ).createInstanceWithValues(values);

          // The base class `values` getter (not overridden here):
          expect(signal.values, equals(values));
          expect(signal.valuesAsDouble, equals(values));
          expect(
            signal.valuesAsString,
            equals(values.map((v) => '$v').toList()),
          );
        });

        test('arithmetic keeps the length', () {
          var values = List<double>.generate(
            2 * blockSize + 1,
            (i) => (i + 1).toDouble(),
          );

          var a = ListSignal(blockSize, 0).createInstanceWithValues(values);
          var b = ListSignal(blockSize, 0).createInstanceWithValues(values);

          expect(a.multiply(b).values, equals(values.map((v) => v * v)));
          expect(
            a.subtract(b).values,
            equals(List<double>.filled(values.length, 0.0)),
          );

          var full = a.createEntryFullOf(2.0);
          expect(
            a.multiplyEntries(full).values,
            equals(values.map((v) => v * 2)),
          );
        });
      });
    }

    test('createEntryN rejects more values than the block size', () {
      var block1 = ListSignal(1, 4);
      expect(() => block1.createEntry2(1, 2), throwsA(isA<UnsupportedError>()));
      expect(
        () => block1.createEntry3(1, 2, 3),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => block1.createEntry4(1, 2, 3, 4),
        throwsA(isA<UnsupportedError>()),
      );

      var block2 = ListSignal(2, 4);
      expect(
        () => block2.createEntry3(1, 2, 3),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => block2.createEntry4(1, 2, 3, 4),
        throwsA(isA<UnsupportedError>()),
      );

      var block3 = ListSignal(3, 4);
      expect(
        () => block3.createEntry4(1, 2, 3, 4),
        throwsA(isA<UnsupportedError>()),
      );
    });

    test('the unsupported operation message names the block size', () {
      var block1 = ListSignal(1, 4);

      expect(
        () => block1.createEntry2(1, 2),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            allOf(contains('2 value'), contains('Entry block size: 1')),
          ),
        ),
      );

      expect(
        () => block1.createEntry3(1, 2, 3),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('3 value'),
          ),
        ),
      );

      expect(
        () => block1.createEntry4(1, 2, 3, 4),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('4 value'),
          ),
        ),
      );
    });

    test('the base entry factories reject every arity', () {
      // An implementation that does not provide its own entry factories:
      var signal = ListSignal(4, 4, useBaseEntryFactories: true);

      expect(() => signal.createEntry1(1), throwsA(isA<UnsupportedError>()));
      expect(() => signal.createEntry2(1, 2), throwsA(isA<UnsupportedError>()));
      expect(
        () => signal.createEntry3(1, 2, 3),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => signal.createEntry4(1, 2, 3, 4),
        throwsA(isA<UnsupportedError>()),
      );
      expect(
        () => signal.createEntryFrom([0, 0, 0, 0], 1),
        throwsA(isA<UnsupportedError>()),
      );

      expect(
        () => signal.createEntry1(1),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            allOf(contains('1 value'), contains('Entry block size: 4')),
          ),
        ),
      );
    });

    test('createEntry dispatches by the number of values', () {
      var block4 = ListSignal(4, 4);

      expect(block4.createEntry([1, 2, 3, 4]), equals([1, 2, 3, 4]));
      expect(block4.createEntry([1, 2, 3]), equals([1, 2, 3, 0]));
      expect(block4.createEntry([1, 2]), equals([1, 2, 0, 0]));
      expect(block4.createEntry([1]), equals([1, 0, 0, 0]));
    });

    test('createInstanceFullOfValue and setAllEntriesWithValue', () {
      var signal = ListSignal(3, 0).createInstanceFullOfValue(5, 7.0);

      expect(signal.length, equals(5));
      expect(signal.values, equals(List<double>.filled(5, 7.0)));

      signal.setAllEntriesEmpty();
      expect(signal.values, equals(List<double>.filled(5, 0.0)));
    });

    test('copy and equality of a custom implementation', () {
      var signal = ListSignal(3, 0).createInstanceWithValues([1, 2, 3, 4, 5]);
      var copy = signal.copy();

      expect(copy.values, equals(signal.values));

      copy.setValue(0, 99.0);
      expect(signal.getValue(0), equals(1.0));
    });

    test('statistics and errors of a custom implementation', () {
      var signal = ListSignal(3, 0).createInstanceWithValues([1, 2, 3, 4]);

      expect(signal.statistics.mean, equals(2.5));
      expect(signal.errorGlobalMean([0, 0, 0, 0]), equals(2.5));
      expect(signal.diff([1, 1, 1, 1]), equals([0, 1, 2, 3]));
    });

    test('normalize of a custom implementation', () {
      var signal = ListSignal(2, 0).createInstanceWithValues([0, 50, 100]);
      var normalized = signal.normalize(ScaleDouble(0, 100));

      expect(normalized.values, equals([0.0, 0.5, 1.0]));
    });

    test('createRandomInstance of a custom implementation', () {
      var signal = ListSignal(
        3,
        0,
      ).createRandomInstance(7, 2.0, dart_math.Random(1));

      expect(signal.length, equals(7));
      expect(signal.values.every((v) => v >= -2 && v <= 2), isTrue);
    });

    test('the generic toString paths', () {
      var signal = ListSignal(2, 0).createInstanceWithValues([1, 2, 3]);

      expect(signal.toString(), isNotEmpty);
      expect(signal.toString(infos: true), contains('length: 3'));
      expect(signal.toString(entries: true), isNotEmpty);
    });
  });

  group('fast_math: remaining paths', () {
    test('expHighPrecision fills the output for the special values', () {
      var out = <double>[0.0, 0.0];

      expect(fast_math.expHighPrecision(double.nan, 0.0, out).isNaN, isTrue);
      expect(out[0].isNaN, isTrue);

      out = <double>[0.0, 0.0];
      expect(
        fast_math.expHighPrecision(double.infinity, 0.0, out),
        equals(double.infinity),
      );
      expect(out[0], equals(double.infinity));

      out = <double>[0.0, 0.0];
      expect(
        fast_math.expHighPrecision(double.negativeInfinity, 0.0, out),
        equals(0.0),
      );
      expect(out[0], equals(0.0));
    });

    test('expHighPrecision handles the subnormal range', () {
      var out = <double>[0.0, 0.0];

      // `intVal > 709` produces a subnormal result:
      var subnormal = fast_math.expHighPrecision(-720.0, 0.0, out);
      expect(subnormal >= 0, isTrue);
      expect(subnormal.isNaN, isFalse);
      expect(subnormal, closeTo(dart_math.exp(-720.0), 1e-310));

      // `intVal == 709` takes a dedicated branch:
      out = <double>[0.0, 0.0];
      var at709 = fast_math.expHighPrecision(-709.5, 0.0, out);
      expect(at709 > 0, isTrue);
      expect(at709, closeTo(dart_math.exp(-709.5), 1e-315));

      // Below the representable range:
      out = <double>[0.0, 0.0];
      expect(fast_math.expHighPrecision(-800.0, 0.0, out), equals(0.0));
      expect(out[0], equals(0.0));
    });

    test('expHighPrecision fills the output on overflow', () {
      var out = <double>[0.0, 0.0];

      expect(fast_math.expHighPrecision(1000.0, 0.0, out).isInfinite, isTrue);
      expect(out[0], equals(double.infinity));
    });

    test('atan of a signed zero on the left plane', () {
      expect(fast_math.atan(0.0, 0.0, true), equals(dart_math.pi));
      expect(fast_math.atan(-0.0, 0.0, true), equals(-dart_math.pi));

      expect(fast_math.atan(0.0, 0.0, false), equals(0.0));
      expect(fast_math.atan(-0.0), equals(-0.0));
    });

    test('atan2 with a zero ordinate and a signed abscissa', () {
      expect(fast_math.atan2(0.0, 1.0), equals(dart_math.atan2(0.0, 1.0)));
      expect(fast_math.atan2(-0.0, 1.0), equals(dart_math.atan2(-0.0, 1.0)));

      expect(
        fast_math.atan2(0.0, -1.0),
        closeTo(dart_math.atan2(0.0, -1.0), 1e-15),
      );
      expect(
        fast_math.atan2(-0.0, -1.0),
        closeTo(dart_math.atan2(-0.0, -1.0), 1e-15),
      );

      // X infinite with a zero ordinate:
      expect(fast_math.atan2(0.0, double.infinity), equals(0.0));
      expect(
        fast_math.atan2(0.0, double.negativeInfinity),
        closeTo(dart_math.pi, 1e-15),
      );
      expect(
        fast_math.atan2(-0.0, double.negativeInfinity),
        closeTo(-dart_math.pi, 1e-15),
      );
    });

    test('atan2 with an infinite abscissa and a signed ordinate', () {
      expect(fast_math.atan2(1.0, double.infinity), equals(0.0));
      expect(fast_math.atan2(-1.0, double.infinity), equals(-0.0));

      expect(
        fast_math.atan2(1.0, double.negativeInfinity),
        closeTo(dart_math.pi, 1e-15),
      );
      expect(
        fast_math.atan2(-1.0, double.negativeInfinity),
        closeTo(-dart_math.pi, 1e-15),
      );
    });
  });
}
