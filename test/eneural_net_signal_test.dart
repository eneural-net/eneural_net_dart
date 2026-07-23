import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  group('Signal: block math', () {
    test('calcNeededBlocks', () {
      expect(Signal.calcNeededBlocks(0, 4), equals(0));
      expect(Signal.calcNeededBlocks(1, 4), equals(1));
      expect(Signal.calcNeededBlocks(4, 4), equals(1));
      expect(Signal.calcNeededBlocks(5, 4), equals(2));
      expect(Signal.calcNeededBlocks(8, 4), equals(2));
      expect(Signal.calcNeededBlocks(9, 4), equals(3));
    });

    test('calcNeededBlocksChunks', () {
      // chunks:1 -> no extra rounding.
      expect(Signal.calcNeededBlocksChunks(5, 4, 1), equals(2));
      expect(Signal.calcNeededBlocksChunks(0, 4, 1), equals(0));

      // chunks:4 -> entries rounded up to a multiple of 4.
      expect(Signal.calcNeededBlocksChunks(1, 4, 4), equals(4));
      expect(Signal.calcNeededBlocksChunks(16, 4, 4), equals(4));
      expect(Signal.calcNeededBlocksChunks(17, 4, 4), equals(8));
      expect(Signal.calcNeededBlocksChunks(0, 4, 4), equals(0));
    });
  });

  group('Signal: structural invariants', () {
    /// Every implementation must keep `lastEntryLength` inside a valid block
    /// and `valuesEntriesLength` covering exactly the values.
    void checkInvariants(Signal signal) {
      var length = signal.length;
      var blockSize = signal.entryBlockSize;

      expect(
        signal.valuesEntriesLength,
        equals(Signal.calcNeededBlocks(length, blockSize)),
        reason: 'valuesEntriesLength of $signal',
      );

      expect(
        signal.valuesEntriesLength <= signal.entriesLength,
        isTrue,
        reason: 'valuesEntriesLength must fit in entriesLength',
      );

      expect(signal.capacity, equals(signal.entriesLength * blockSize));
      expect(signal.capacity >= length, isTrue);

      if (length == 0) {
        expect(signal.lastEntryLength, equals(0));
      } else {
        expect(signal.lastEntryLength >= 1, isTrue);
        expect(signal.lastEntryLength <= blockSize, isTrue);
        expect(
          (signal.valuesEntriesLength - 1) * blockSize + signal.lastEntryLength,
          equals(length),
          reason: 'lastEntryLength must complete the length of $signal',
        );
      }
    }

    test('SignalFloat32x4 for sizes 0..64', () {
      for (var i = 0; i <= 64; ++i) {
        checkInvariants(SignalFloat32x4(i));
        checkInvariants(
          SignalFloat32x4.from(List<double>.generate(i, (i) => i.toDouble())),
        );
      }
    });

    test('SignalInt32x4 for sizes 0..64', () {
      for (var i = 0; i <= 64; ++i) {
        checkInvariants(SignalInt32x4(i));
        checkInvariants(SignalInt32x4.from(List<int>.generate(i, (i) => i)));
      }
    });

    test('SignalFloat32x4Mod4 for sizes 0..64', () {
      for (var i = 0; i <= 64; ++i) {
        checkInvariants(SignalFloat32x4Mod4(i));
        checkInvariants(
          SignalFloat32x4Mod4.from(
            List<double>.generate(i, (i) => i.toDouble()),
          ),
        );
      }
    });

    test('the size constructor and `from` agree structurally', () {
      for (var i = 0; i <= 32; ++i) {
        var zeroesInt = List<int>.filled(i, 0);
        expect(
          SignalInt32x4(i).entriesLength,
          equals(SignalInt32x4.from(zeroesInt).entriesLength),
          reason: 'SignalInt32x4 of size $i',
        );
        expect(SignalInt32x4(i), equals(SignalInt32x4.from(zeroesInt)));

        var zeroesDouble = List<double>.filled(i, 0);
        expect(
          SignalFloat32x4(i).entriesLength,
          equals(SignalFloat32x4.from(zeroesDouble).entriesLength),
          reason: 'SignalFloat32x4 of size $i',
        );
        expect(
          SignalFloat32x4Mod4(i).entriesLength,
          equals(SignalFloat32x4Mod4.from(zeroesDouble).entriesLength),
          reason: 'SignalFloat32x4Mod4 of size $i',
        );
      }
    });

    test('SIMD chunked implementations allocate entries in chunks of 4', () {
      // Their operation loops are unrolled 4 entries at a time.
      for (var i = 1; i <= 40; ++i) {
        expect(SignalInt32x4(i).entriesLength % 4, equals(0));
        expect(
          SignalInt32x4.from(List<int>.filled(i, 1)).entriesLength % 4,
          equals(0),
        );
        expect(SignalFloat32x4Mod4(i).entriesLength % 4, equals(0));
        expect(
          SignalFloat32x4Mod4.from(List<double>.filled(i, 1)).entriesLength % 4,
          equals(0),
        );
      }
    });
  });

  group('Signal: values access', () {
    test('getValue/setValue/addToValue', () {
      var s = SignalFloat32x4(6);

      expect(s.values, equals([0, 0, 0, 0, 0, 0]));

      s.setValue(0, 1.0);
      s.setValue(3, 4.0);
      s.setValue(5, 6.0);

      expect(s.getValue(0), equals(1.0));
      expect(s.getValue(3), equals(4.0));
      expect(s.getValue(5), equals(6.0));
      expect(s.values, equals([1, 0, 0, 4, 0, 6]));

      s.addToValue(0, 10.0);
      expect(s.getValue(0), equals(11.0));
    });

    test('operator [] and []= delegate to getValue/setValue', () {
      var s = SignalFloat32x4.from([1, 2, 3]);
      expect(s[1], equals(2.0));
      s[1] = 20.0;
      expect(s.getValue(1), equals(20.0));
    });

    test('getValueEntryIndex', () {
      var s = SignalFloat32x4(9);
      expect(s.getValueEntryIndex(0), equals(0));
      expect(s.getValueEntryIndex(3), equals(0));
      expect(s.getValueEntryIndex(4), equals(1));
      expect(s.getValueEntryIndex(8), equals(2));
    });

    test('getValues clamps the requested length', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      expect(s.getValues(), equals([1, 2, 3, 4, 5]));
      expect(s.getValues(3), equals([1, 2, 3]));
      expect(s.getValues(50), equals([1, 2, 3, 4, 5]));
      expect(s.getValues(0), isEmpty);
      expect(s.getValues(-1), isEmpty);
    });

    test('getEntries clamps the requested length', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      expect(s.getEntries().length, equals(s.entriesLength));
      expect(s.getEntries(1).length, equals(1));
      expect(s.getEntries(50).length, equals(s.entriesLength));
      expect(s.getEntries(0), isEmpty);
      expect(s.getEntries(-1), isEmpty);
    });

    test('valuesAsDouble/valuesAsString', () {
      var si = SignalInt32x4.from([1, 2, 3]);
      expect(si.valuesAsDouble, equals([1.0, 2.0, 3.0]));
      expect(si.valuesAsString, equals(['1', '2', '3']));

      var sf = SignalFloat32x4.from([1.5, 2.5]);
      expect(sf.valuesAsDouble, equals([1.5, 2.5]));
      expect(sf.valuesAsString.length, equals(2));
    });

    test('empty signal has no values', () {
      var s = SignalFloat32x4(0);
      expect(s.length, equals(0));
      expect(s.values, isEmpty);
      expect(s.isEmpty, isTrue);
      expect(SignalInt32x4(0).values, isEmpty);
    });
  });

  group('Signal: as a fixed-length List', () {
    test('is a List of its values', () {
      var s = SignalFloat32x4.from([1, 2, 3]);
      expect(s, isA<List<double>>());
      expect(s.length, equals(3));
      expect(s.contains(2.0), isTrue);
      expect(s.toList(), equals([1.0, 2.0, 3.0]));
      expect(s.reduce((a, b) => a + b), equals(6.0));
    });

    test('cannot grow or shrink', () {
      var s = SignalFloat32x4.from([1, 2, 3]);
      expect(() => s.add(4.0), throwsA(isA<UnsupportedError>()));
      expect(() => s.length = 5, throwsA(isA<UnsupportedError>()));
    });
  });

  group('Signal: entries', () {
    test('getEntry/setEntry/addToEntry/subtractToEntry (Float32x4)', () {
      var s = SignalFloat32x4(4);

      s.setEntry(0, Float32x4(1, 2, 3, 4));
      expect(s.values, equals([1, 2, 3, 4]));

      s.addToEntry(0, Float32x4(1, 1, 1, 1));
      expect(s.values, equals([2, 3, 4, 5]));

      s.subtractToEntry(0, Float32x4(2, 2, 2, 2));
      expect(s.values, equals([0, 1, 2, 3]));
    });

    test('getEntry/setEntry/addToEntry/subtractToEntry (Int32x4)', () {
      var s = SignalInt32x4(4);

      s.setEntry(0, Int32x4(1, 2, 3, 4));
      expect(s.values, equals([1, 2, 3, 4]));

      s.addToEntry(0, Int32x4(1, 1, 1, 1));
      expect(s.values, equals([2, 3, 4, 5]));

      s.subtractToEntry(0, Int32x4(2, 2, 2, 2));
      expect(s.values, equals([0, 1, 2, 3]));
    });

    test('getEntryFiltered/setEntryFiltered', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);

      var filtered = s.getEntryFiltered(0, (n) => n * 2);
      expect(filtered.x, equals(2.0));
      expect(s.values, equals([1, 2, 3, 4]), reason: 'get must not mutate');

      s.setEntryFiltered(0, (n) => n * 2);
      expect(s.values, equals([2, 4, 6, 8]));
    });

    test('getEntryFilteredX4/setEntryFilteredX4', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);

      s.setEntryFilteredX4(0, (e) => e + Float32x4.splat(1));
      expect(s.values, equals([2, 3, 4, 5]));

      var si = SignalInt32x4.from([1, 2, 3, 4]);
      si.setEntryFilteredX4(0, (e) => e + Int32x4(1, 1, 1, 1));
      expect(si.values, equals([2, 3, 4, 5]));
    });

    test('setEntryWithValue/setEntryEmpty', () {
      var s = SignalFloat32x4(4);
      s.setEntryWithValue(0, 7.0);
      expect(s.values, equals([7, 7, 7, 7]));

      s.setEntryEmpty(0);
      expect(s.values, equals([0, 0, 0, 0]));
    });

    test('setAllEntriesWithValue/setAllEntriesEmpty', () {
      var s = SignalFloat32x4(8);
      s.setAllEntriesWithValue(3.0);
      expect(s.values, equals(List.filled(8, 3.0)));

      s.setAllEntriesEmpty();
      expect(s.values, equals(List.filled(8, 0.0)));
    });

    test('setAllEntriesWith copies another signal', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4]);
      var b = SignalFloat32x4(4);

      b.setAllEntriesWith(a);
      expect(b.values, equals([1, 2, 3, 4]));
    });

    test('set copies a limited number of entries', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4, 5, 6, 7, 8]);
      var b = SignalFloat32x4(8);

      b.set(a, 1);
      expect(b.values, equals([1, 2, 3, 4, 0, 0, 0, 0]));

      b.set(a);
      expect(b.values, equals([1, 2, 3, 4, 5, 6, 7, 8]));
    });

    test('setEntryValues1..4', () {
      var s = SignalFloat32x4(4);

      s.setEntryValues1(0, 1.0);
      expect(s.values, equals([1, 0, 0, 0]));

      s.setEntryValues2(0, 1.0, 2.0);
      expect(s.values, equals([1, 2, 0, 0]));

      s.setEntryValues3(0, 1.0, 2.0, 3.0);
      expect(s.values, equals([1, 2, 3, 0]));

      s.setEntryValues4(0, 1.0, 2.0, 3.0, 4.0);
      expect(s.values, equals([1, 2, 3, 4]));
    });

    test('setEntryWithRandomValues uses the given Random', () {
      var s = SignalFloat32x4(4);
      s.setEntryWithRandomValues(0, 10.0, Random(123));

      var s2 = SignalFloat32x4(4);
      s2.setEntryWithRandomValues(0, 10.0, Random(123));

      expect(s.values, equals(s2.values));
      expect(s.values.every((v) => v >= -10 && v <= 10), isTrue);
    });

    test('createEntry from a values list', () {
      var s = SignalFloat32x4(4);

      expect(s.createEntry([1.0]).x, equals(1.0));
      expect(s.createEntry([1.0, 2.0]).y, equals(2.0));
      expect(s.createEntry([1.0, 2.0, 3.0]).z, equals(3.0));
      expect(s.createEntry([1.0, 2.0, 3.0, 4.0]).w, equals(4.0));

      expect(() => s.createEntry([]), throwsA(isA<StateError>()));
      expect(
        () => s.createEntry([1.0, 2.0, 3.0, 4.0, 5.0]),
        throwsA(isA<StateError>()),
      );
    });

    test('createEntryFrom keeps the non-overridden lanes', () {
      var s = SignalFloat32x4(4);
      var base = Float32x4(1, 2, 3, 4);

      var e = s.createEntryFrom(base, null, 20.0, null, 40.0);
      expect([e.x, e.y, e.z, e.w], equals([1.0, 20.0, 3.0, 40.0]));

      var si = SignalInt32x4(4);
      var ei = si.createEntryFrom(Int32x4(1, 2, 3, 4), 10, null, null, null);
      expect([ei.x, ei.y, ei.z, ei.w], equals([10, 2, 3, 4]));
    });

    test('createEntryFullOf/entryEmpty', () {
      var s = SignalFloat32x4(4);
      var e = s.createEntryFullOf(5.0);
      expect([e.x, e.y, e.z, e.w], equals([5.0, 5.0, 5.0, 5.0]));

      var empty = s.entryEmpty;
      expect([empty.x, empty.y, empty.z, empty.w], equals([0, 0, 0, 0]));
    });

    test('getValueFromEntry/setValueFromEntry reject invalid offsets', () {
      var s = SignalFloat32x4(4);
      var e = Float32x4(1, 2, 3, 4);

      for (var i = 0; i < 4; ++i) {
        expect(s.getValueFromEntry(e, i), equals(i + 1.0));
      }

      expect(() => s.getValueFromEntry(e, 4), throwsA(isA<StateError>()));
      expect(() => s.getValueFromEntry(e, -1), throwsA(isA<StateError>()));
      expect(() => s.setValueFromEntry(e, 4, 0.0), throwsA(isA<StateError>()));
      expect(() => s.addValueFromEntry(e, 4, 0.0), throwsA(isA<StateError>()));

      var si = SignalInt32x4(4);
      var ei = Int32x4(1, 2, 3, 4);
      expect(() => si.getValueFromEntry(ei, 9), throwsA(isA<StateError>()));
      expect(() => si.setValueFromEntry(ei, 9, 0), throwsA(isA<StateError>()));
      expect(() => si.addValueFromEntry(ei, 9, 0), throwsA(isA<StateError>()));
    });

    test('addValueFromEntry adds to the selected lane', () {
      var s = SignalFloat32x4(4);
      var e = Float32x4(1, 2, 3, 4);

      expect(s.addValueFromEntry(e, 0, 10.0).x, equals(11.0));
      expect(s.addValueFromEntry(e, 1, 10.0).y, equals(12.0));
      expect(s.addValueFromEntry(e, 2, 10.0).z, equals(13.0));
      expect(s.addValueFromEntry(e, 3, 10.0).w, equals(14.0));
    });
  });

  group('Signal: extra values (padding)', () {
    test('setExtraValuesToZero/One on a non-chunked signal', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      expect(s.capacity, equals(8));

      s.setExtraValuesToOne();
      expect(s.values, equals([1, 2, 3, 4, 5]), reason: 'values are untouched');
      expect(s.getEntry(1).y, equals(1.0));
      expect(s.getEntry(1).z, equals(1.0));
      expect(s.getEntry(1).w, equals(1.0));

      s.setExtraValuesToZero();
      expect(s.getEntry(1).y, equals(0.0));
    });

    test('setExtraValues fills the chunk padding too', () {
      // 5 values -> 2 used entries, but 4 allocated entries (chunk of 4).
      var s = SignalInt32x4.from([1, 2, 3, 4, 5]);
      expect(s.valuesEntriesLength, equals(2));
      expect(s.entriesLength, equals(4));

      s.setExtraValues(9);

      expect(s.values, equals([1, 2, 3, 4, 5]));

      // Tail of the last used entry:
      expect(s.getEntry(1).y, equals(9));
      // Entirely unused entries:
      expect(s.getEntry(2).x, equals(9));
      expect(s.getEntry(3).w, equals(9));
    });

    test('setExtraValues is a no-op when there is no padding', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);
      expect(s.capacity, equals(s.length));

      s.setExtraValues(9.0);
      expect(s.values, equals([1, 2, 3, 4]));
    });

    test('setExtraValues on an empty signal does not throw', () {
      expect(() => SignalFloat32x4(0).setExtraValuesToZero(), returnsNormally);
      expect(() => SignalInt32x4(0).setExtraValuesToZero(), returnsNormally);
    });

    test('setExtraValues works for every size of every implementation', () {
      for (var i = 0; i <= 40; ++i) {
        expect(
          () => SignalFloat32x4(i).setExtraValuesToZero(),
          returnsNormally,
          reason: 'SignalFloat32x4($i)',
        );
        expect(
          () => SignalInt32x4(i).setExtraValuesToZero(),
          returnsNormally,
          reason: 'SignalInt32x4($i)',
        );
        expect(
          () => SignalFloat32x4Mod4(i).setExtraValuesToZero(),
          returnsNormally,
          reason: 'SignalFloat32x4Mod4($i)',
        );
      }
    });
  });

  group('Signal: arithmetic', () {
    test('multiply/multiplyTo (Float32x4)', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      var b = SignalFloat32x4.from([2, 2, 2, 2, 2]);

      var c = a.multiply(b);
      expect(c.length, equals(a.length), reason: 'keeps the operand length');
      expect(c.values, equals([2, 4, 6, 8, 10]));

      var d = SignalFloat32x4(5);
      a.multiplyTo(b, d);
      expect(d.values, equals([2, 4, 6, 8, 10]));
    });

    test('subtract/subtractTo (Float32x4)', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      var b = SignalFloat32x4.from([1, 1, 1, 1, 1]);

      var c = a.subtract(b);
      expect(c.length, equals(5));
      expect(c.values, equals([0, 1, 2, 3, 4]));

      var d = SignalFloat32x4(5);
      a.subtractTo(b, d);
      expect(d.values, equals([0, 1, 2, 3, 4]));
    });

    test('multiply/subtract (Int32x4)', () {
      var a = SignalInt32x4.from([1, 2, 3, 4, 5]);
      var b = SignalInt32x4.from([2, 2, 2, 2, 2]);

      expect(a.multiply(b).values, equals([2, 4, 6, 8, 10]));
      expect(a.subtract(b).values, equals([-1, 0, 1, 2, 3]));
    });

    test('multiply/subtract (Float32x4Mod4)', () {
      var a = SignalFloat32x4Mod4.from([1, 2, 3, 4, 5]);
      var b = SignalFloat32x4Mod4.from([2, 2, 2, 2, 2]);

      var c = a.multiply(b);
      expect(c, isA<SignalFloat32x4Mod4>());
      expect(c.values, equals([2, 4, 6, 8, 10]));
      expect(a.subtract(b).values, equals([-1, 0, 1, 2, 3]));
    });

    test('multiplyEntries/multiplyAllEntriesTo', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4]);

      var c = a.multiplyEntries(Float32x4.splat(3));
      expect(c.length, equals(4));
      expect(c.values, equals([3, 6, 9, 12]));

      var d = SignalFloat32x4(4);
      a.multiplyAllEntriesTo(Float32x4.splat(2), d);
      expect(d.values, equals([2, 4, 6, 8]));
    });

    test('subtractAllEntriesTo', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4]);
      var d = SignalFloat32x4(4);

      a.subtractAllEntriesTo(Float32x4.splat(1), d);
      expect(d.values, equals([0, 1, 2, 3]));
    });

    test('multiplyValueTo/multiplyAllValuesAddingTo', () {
      var a = SignalFloat32x4.from([1, 2, 3, 4]);

      var d = SignalFloat32x4(4);
      a.multiplyValueTo(2.0, d);
      expect(d.values, equals([2, 4, 6, 8]));

      // Accumulates into the destiny:
      a.multiplyAllValuesAddingTo(2.0, d);
      expect(d.values, equals([4, 8, 12, 16]));
    });

    test('entry operations (Float32x4)', () {
      var s = SignalFloat32x4(4);
      var e1 = Float32x4(4, 6, 8, 10);
      var e2 = Float32x4(2, 2, 2, 2);

      expect(s.entryOperationSum(e1, e2).x, equals(6.0));
      expect(s.entryOperationSubtract(e1, e2).x, equals(2.0));
      expect(s.entryOperationMultiply(e1, e2).x, equals(8.0));
      expect(s.entryOperationDivide(e1, e2).x, equals(2.0));

      expect(s.entryOperationSumLane(e1), equals(28.0));
      expect(s.entryOperationSumLanePartial(e1, 2), equals(10.0));
      expect(s.entryOperationSumSquaresLane(e2), equals(16.0));
      expect(s.entryOperationSumSquaresLanePartial(e2, 2), equals(8.0));
    });

    test('entry operations (Int32x4)', () {
      var s = SignalInt32x4(4);
      var e1 = Int32x4(4, 6, 8, 10);
      var e2 = Int32x4(2, 2, 2, 2);

      expect(s.entryOperationSum(e1, e2).x, equals(6));
      expect(s.entryOperationSubtract(e1, e2).x, equals(2));
      expect(s.entryOperationMultiply(e1, e2).x, equals(8));
      expect(s.entryOperationDivide(e1, e2).x, equals(2));

      expect(s.entryOperationSumLane(e1), equals(28));
      expect(s.entryOperationSumLanePartial(e1, 3), equals(18));
      expect(s.entryOperationSumSquaresLane(e2), equals(16));
      expect(s.entryOperationSumSquaresLanePartial(e2, 3), equals(12));
    });
  });

  group('Signal: statistics and errors', () {
    test('computeSumSquares/computeSumSquaresMean (Float32x4)', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4, 5]);
      expect(s.computeSumSquares(), equals(55.0));
      expect(s.computeSumSquaresMean(), equals(11.0));
    });

    test('computeSumSquares ignores the chunk padding (Int32x4)', () {
      var s = SignalInt32x4.from([1, 2, 3, 4, 5]);
      expect(s.entriesLength, greaterThan(s.valuesEntriesLength));
      expect(s.computeSumSquares(), equals(55));
      expect(s.computeSumSquaresMean(), equals(11.0));
    });

    test('computeSumSquares ignores the chunk padding (Float32x4Mod4)', () {
      var s = SignalFloat32x4Mod4.from([1, 2, 3, 4, 5]);
      expect(s.computeSumSquares(), equals(55.0));
      expect(s.computeSumSquaresMean(), equals(11.0));
    });

    test('computeSumSquares for every size', () {
      for (var i = 1; i <= 33; ++i) {
        var values = List<double>.generate(i, (i) => (i + 1).toDouble());
        var expected = values.fold<double>(0, (t, v) => t + (v * v));

        expect(
          SignalFloat32x4.from(values).computeSumSquares(),
          closeTo(expected, 1e-6),
          reason: 'SignalFloat32x4 of size $i',
        );
        expect(
          SignalFloat32x4Mod4.from(values).computeSumSquares(),
          closeTo(expected, 1e-6),
          reason: 'SignalFloat32x4Mod4 of size $i',
        );
        expect(
          SignalInt32x4.from(
            values.map((e) => e.toInt()).toList(),
          ).computeSumSquares(),
          equals(expected.toInt()),
          reason: 'SignalInt32x4 of size $i',
        );
      }
    });

    test('computeSumSquares of an empty signal is zero', () {
      expect(SignalFloat32x4(0).computeSumSquares(), equals(0.0));
      expect(SignalInt32x4(0).computeSumSquares(), equals(0));
    });

    test('statistics', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);
      var st = s.statistics;

      expect(st.length, equals(4));
      expect(st.min, equals(1.0));
      expect(st.max, equals(4.0));
      expect(st.mean, equals(2.5));
    });

    test('diff/diffAbs/errors/errorsAbs', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);

      expect(s.diff([0, 4, 0, 4]), equals([1, -2, 3, 0]));
      expect(s.diffAbs([0, 4, 0, 4]), equals([1, 2, 3, 0]));
      expect(s.errors([0, 4, 0, 4]), equals([1, -2, 3, 0]));
      expect(s.errorsAbs([0, 4, 0, 4]), equals([1, 2, 3, 0]));
    });

    test('errorGlobalMean/SquareMean/SquareMeanRoot', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4]);
      var expected = [0.0, 0.0, 0.0, 0.0];

      expect(s.errorGlobalMean(expected), equals(2.5));
      expect(s.errorGlobalSquareMean(expected), equals(30 / 4));
      expect(
        s.errorGlobalSquareMeanRoot(expected),
        closeTo(sqrt(30 / 4), 1e-12),
      );
    });

    test('normalize applies the scale to every value', () {
      var s = SignalFloat32x4.from([0, 50, 100]);
      var normalized = s.normalize(ScaleDouble(0, 100));

      expect(normalized.length, equals(3));
      expect(normalized.values, equals([0.0, 0.5, 1.0]));
      expect(s.values, equals([0, 50, 100]), reason: 'the source is unchanged');
    });

    test('normalizeEntry', () {
      var s = SignalFloat32x4(4);
      var e = s.normalizeEntry(Float32x4(0, 50, 100, 200), ScaleDouble(0, 100));
      expect([e.x, e.y, e.z, e.w], equals([0.0, 0.5, 1.0, 2.0]));

      var si = SignalInt32x4(4);
      var ei = si.normalizeEntry(Int32x4(0, 50, 100, 200), ScaleInt(0, 100));
      expect([ei.x, ei.y, ei.z, ei.w], equals([0, 0, 1, 2]));
    });
  });

  group('Signal: copy and instances', () {
    test('copy is an independent equal instance', () {
      for (var s in <Signal>[
        SignalFloat32x4.from([1, 2, 3, 4, 5]),
        SignalInt32x4.from([1, 2, 3, 4, 5]),
        SignalFloat32x4Mod4.from([1, 2, 3, 4, 5]),
      ]) {
        var copy = s.copy();

        expect(copy.runtimeType, equals(s.runtimeType));
        expect(copy, equals(s));
        expect(copy.values, equals(s.values));

        copy.setValue(0, copy.toN(99));
        expect(s.getValue(0), isNot(equals(99)), reason: '${s.runtimeType}');
      }
    });

    test('createInstance/createInstanceOfSameLength', () {
      var s = SignalFloat32x4.from([1, 2, 3, 4, 5]);

      var i1 = s.createInstance(3);
      expect(i1.length, equals(3));
      expect(i1.values, equals([0, 0, 0]));

      var i2 = s.createInstanceOfSameLength();
      expect(i2.length, equals(5));

      var i3 = s.createInstanceOfSameLengthFullOfValue(7.0);
      expect(i3.values, equals([7, 7, 7, 7, 7]));

      var i4 = s.createInstanceFullOfValue(3, 2.0);
      expect(i4.values, equals([2, 2, 2]));
    });

    test('createInstance keeps the concrete implementation', () {
      expect(
        SignalFloat32x4Mod4(4).createInstance(4),
        isA<SignalFloat32x4Mod4>(),
      );
      expect(SignalInt32x4(4).createInstance(4), isA<SignalInt32x4>());
    });

    test('createRandomInstance is bounded by the scale', () {
      var s = SignalFloat32x4(1).createRandomInstance(8, 5.0, Random(7));
      expect(s.length, equals(8));
      expect(s.values.every((v) => v >= -5 && v <= 5), isTrue);

      var si = SignalInt32x4(1).createRandomInstance(8, 5, Random(7));
      expect(si.length, equals(8));
      expect(si.values.every((v) => v >= -5 && v <= 5), isTrue);
    });

    test('createRandomValue/createRandomEntry are bounded', () {
      var s = SignalFloat32x4(4);
      for (var i = 0; i < 50; ++i) {
        var v = s.createRandomValue(3.0);
        expect(v >= -3 && v <= 3, isTrue);
      }

      var e = s.createRandomEntry(3.0, Random(1));
      expect([e.x, e.y, e.z, e.w].every((v) => v >= -3 && v <= 3), isTrue);

      var si = SignalInt32x4(4);
      var ei = si.createRandomEntry(3, Random(1));
      expect([ei.x, ei.y, ei.z, ei.w].every((v) => v >= -3 && v <= 3), isTrue);
    });

    test('createRandomEntries', () {
      var entries = SignalFloat32x4(4).createRandomEntries(6, 2.0, Random(1));
      expect(entries.length, equals(6));
    });

    test('valuesToEntries handles every tail size', () {
      var s = SignalFloat32x4(4);

      expect(s.valuesToEntries([]).length, equals(0));
      expect(s.valuesToEntries([1]).length, equals(1));
      expect(s.valuesToEntries([1, 2]).length, equals(1));
      expect(s.valuesToEntries([1, 2, 3]).length, equals(1));
      expect(s.valuesToEntries([1, 2, 3, 4]).length, equals(1));
      expect(s.valuesToEntries([1, 2, 3, 4, 5]).length, equals(2));

      // Unused lanes of the tail entry are zero:
      var tail = s.valuesToEntries([1, 2, 3, 4, 5])[1];
      expect([tail.x, tail.y, tail.z, tail.w], equals([5.0, 0.0, 0.0, 0.0]));
    });
  });

  group('Signal: equality', () {
    test('same values and length are equal', () {
      expect(
        SignalFloat32x4.from([1, 2, 3]),
        equals(SignalFloat32x4.from([1, 2, 3])),
      );
      expect(
        SignalFloat32x4.from([1, 2, 3]).hashCode,
        equals(SignalFloat32x4.from([1, 2, 3]).hashCode),
      );
      expect(
        SignalInt32x4.from([1, 2, 3]),
        equals(SignalInt32x4.from([1, 2, 3])),
      );
      expect(
        SignalInt32x4.from([1, 2, 3]).hashCode,
        equals(SignalInt32x4.from([1, 2, 3]).hashCode),
      );
      expect(
        SignalFloat32x4Mod4.from([1, 2, 3]).hashCode,
        equals(SignalFloat32x4Mod4.from([1, 2, 3]).hashCode),
      );
    });

    test('equal signals can be used as Set/Map keys', () {
      // `hashCode` must follow `==`, otherwise equal signals end up in
      // different buckets.
      var set = <Signal>{
        SignalFloat32x4.from([1, 2, 3]),
        SignalFloat32x4.from([1, 2, 3]),
        SignalInt32x4.from([1, 2, 3]),
        SignalInt32x4.from([1, 2, 3]),
      };

      expect(set.length, equals(2));

      var map = <Signal, String>{};
      map[SignalFloat32x4.from([1, 2])] = 'a';
      map[SignalFloat32x4.from([1, 2])] = 'b';

      expect(map.length, equals(1));
      expect(map[SignalFloat32x4.from([1, 2])], equals('b'));
    });

    test('different values or lengths are not equal', () {
      expect(
        SignalFloat32x4.from([1, 2, 3]),
        isNot(equals(SignalFloat32x4.from([1, 2, 4]))),
      );
      expect(
        SignalFloat32x4.from([1, 2, 3]),
        isNot(equals(SignalFloat32x4.from([1, 2, 3, 4]))),
      );
      expect(SignalFloat32x4.from([1]), isNot(equals('not a signal')));
      expect(SignalInt32x4.from([1]), isNot(equals(SignalInt32x4.from([2]))));
    });

    test('identical is equal', () {
      var s = SignalFloat32x4.from([1, 2, 3]);
      expect(s, equals(s));

      var si = SignalInt32x4.from([1, 2, 3]);
      expect(si, equals(si));
    });
  });

  group('Signal: fromFormat', () {
    test('creates by size', () {
      expect(
        Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
          'Float32x4',
          size: 4,
        ),
        isA<SignalFloat32x4>(),
      );
      expect(
        Signal.fromFormat<int, Int32x4, SignalInt32x4>('Int32x4', size: 4),
        isA<SignalInt32x4>(),
      );
      expect(
        Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
          'Float32x4Mod4',
          size: 4,
        ),
        isA<SignalFloat32x4Mod4>(),
      );
    });

    test('creates from values', () {
      var s = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
        'Float32x4',
        values: [1, 2, 3],
      );
      expect(s.values, equals([1, 2, 3]));

      var si = Signal.fromFormat<int, Int32x4, SignalInt32x4>(
        'Int32x4',
        values: [1, 2, 3],
      );
      expect(si.values, equals([1, 2, 3]));

      var sm = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
        'Float32x4Mod4',
        values: [1, 2, 3],
      );
      expect(sm.values, equals([1, 2, 3]));
    });

    test('creates from entries', () {
      var s = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
        'Float32x4',
        size: 4,
        entries: [Float32x4(1, 2, 3, 4)],
      );
      expect(s.values, equals([1, 2, 3, 4]));

      var si = Signal.fromFormat<int, Int32x4, SignalInt32x4>(
        'Int32x4',
        size: 4,
        entries: [Int32x4(1, 2, 3, 4)],
      );
      expect(si.values, equals([1, 2, 3, 4]));

      var sm = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
        'Float32x4Mod4',
        size: 4,
        entries: [Float32x4(1, 2, 3, 4)],
      );
      expect(sm.values, equals([1, 2, 3, 4]));
    });

    test('unknown format throws', () {
      expect(
        () => Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
          'Nope',
          size: 1,
        ),
        throwsA(isA<StateError>()),
      );
    });
  });

  group('Signal: formats and toString', () {
    test('format names', () {
      expect(SignalFloat32x4(1).format, equals('Float32x4'));
      expect(SignalInt32x4(1).format, equals('Int32x4'));
      expect(SignalFloat32x4Mod4(1).format, equals('Float32x4Mod4'));
    });

    test('toString variants', () {
      var s = SignalFloat32x4.from([1, 2, 3]);

      expect(s.toString(), contains('1'));
      expect(s.toString(entries: true), isNotEmpty);
      expect(s.toString(infos: true), contains('length: 3'));
      expect(
        s.toString(infos: true, entries: true),
        contains('SignalFloat32x4'),
      );

      expect(s.toStringWithValues(), contains('length: 3'));
      expect(s.toStringWithEntries(), contains('entries'));
    });

    test('toString truncates long signals', () {
      var s = SignalFloat32x4.from(
        List<double>.generate(100, (i) => i.toDouble()),
      );
      expect(s.toStringValues(10), contains('...[#100]'));
      expect(s.toStringEntries(2), contains('...[#25]'));
    });

    test('nToString', () {
      expect(SignalInt32x4(1).nToString(7), equals('7'));
      expect(SignalFloat32x4(1).nToString(1.5), isNotEmpty);
    });

    test('zero/one/toN', () {
      var sf = SignalFloat32x4(1);
      expect(sf.zero, equals(0.0));
      expect(sf.one, equals(1.0));
      expect(sf.toN(2), equals(2.0));

      var si = SignalInt32x4(1);
      expect(si.zero, equals(0));
      expect(si.one, equals(1));
      expect(si.toN(2.9), equals(2));
    });

    test('calcEntriesCapacityForSize', () {
      expect(SignalFloat32x4(1).calcEntriesCapacityForSize(5), equals(2));
      expect(SignalInt32x4(1).calcEntriesCapacityForSize(5), equals(4));
    });
  });
}
