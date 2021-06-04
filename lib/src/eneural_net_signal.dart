import 'dart:collection';
import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:eneural_net/eneural_net.dart';
import 'package:swiss_knife/swiss_knife.dart' show formatDecimal;

import 'eneural_net_extension.dart';
import 'eneural_net_scale.dart';

class SignalInt32x4 extends Signal<int, Int32x4, SignalInt32x4> {
  static final SignalInt32x4 EMPTY = SignalInt32x4(0);

  final Int32x4List _entries;
  final int _entriesLength;

  final int _size;

  SignalInt32x4(int size)
      : _size = size,
        _entries = Int32x4List(calcEntriesCapacity(size)),
        _entriesLength = calcEntriesCapacity(size) {
    assert(_entries.length == _entriesLength);
  }

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 4);

  SignalInt32x4._(this._entries, this._size)
      : _entriesLength = _entries.length {
    assert(_entries.length == _entriesLength);
  }

  factory SignalInt32x4.from(List<int> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalInt32x4.fromEntries(List<Int32x4> entries, int size) =>
      SignalInt32x4._(Int32x4List.fromList(entries), size);

  @override
  String get format => 'Int32x4';

  @override
  int calcEntriesCapacityForSize(int size) =>
      Signal.calcNeededBlocksChunks(size, 4, 4);

  @override
  SignalInt32x4 copy() =>
      SignalInt32x4._(Int32x4List.fromList(_entries), _size);

  @override
  List<int> get values {
    final lastEntryIndex = entriesLength - 1;
    var entryIndex = 0;
    var entry = _entries.first;
    var entryCursor = -1;
    var valueCursor = -1;

    var list = List.generate(length, (i) {
      assert(++valueCursor == i);

      switch (++entryCursor) {
        case 0:
          {
            return entry.x;
          }
        case 1:
          {
            return entry.y;
          }
        case 2:
          {
            return entry.z;
          }
        case 3:
          {
            var w = entry.w;
            if (entryIndex < lastEntryIndex) {
              entry = _entries[++entryIndex];
              entryCursor = -1;
            }
            return w;
          }
        default:
          throw StateError('Invalid entryCursor: $entryCursor');
      }
    });

    return list;
  }

  @override
  int get zero => 0;

  @override
  int get one => 1;

  @override
  SignalInt32x4 createInstance(int size) => SignalInt32x4(size);

  @override
  SignalInt32x4 createRandomInstance(int size, int randomScale,
      [Random? rand]) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale, rand);
    return SignalInt32x4.fromEntries(entries, size);
  }

  @override
  SignalInt32x4 createInstanceWithEntries(int size, List<Int32x4> entries) {
    ensureEntriesLengthMod(entries);
    return SignalInt32x4._(Int32x4List.fromList(entries), size);
  }

  @override
  void ensureEntriesLengthMod(List<Int32x4> entries) {}

  @override
  int toN(num n) => n.toInt();

  @override
  String nToString(int n) => '$n';

  static final int ENTRY_BLOCK_SIZE = 4;

  @override
  int get entryBlockSize => ENTRY_BLOCK_SIZE;

  @override
  int get length => _size;

  @override
  int get capacity => _entriesLength * 4;

  @override
  int get entriesLength => _entriesLength;

  @override
  List<Int32x4> get entries => _entries.toList();

  @override
  Int32x4 getEntry(int index) => _entries[index];

  @override
  void setEntry(int index, Int32x4 entry) => _entries[index] = entry;

  @override
  void addToEntry(int index, Int32x4 entry) {
    _entries[index] += entry;
  }

  @override
  void subtractToEntry(int index, Int32x4 entry) {
    _entries[index] -= entry;
  }

  @override
  Int32x4 getEntryFilteredX4(
      int index, Int32x4 Function(Int32x4 entry) filter) {
    return filter(_entries[index]);
  }

  @override
  Int32x4 getEntryFiltered(int index, int Function(int n) filter) {
    var entry = _entries[index];
    return Int32x4(
      filter(entry.x),
      filter(entry.y),
      filter(entry.z),
      filter(entry.w),
    );
  }

  @override
  int getValueFromEntry(Int32x4 entry, int offset) {
    switch (offset) {
      case 0:
        return entry.x;
      case 1:
        return entry.y;
      case 2:
        return entry.z;
      case 3:
        return entry.w;
      default:
        throw StateError('Invalid Int32x4 entry offset: $offset/4');
    }
  }

  @override
  Int32x4 setValueFromEntry(Int32x4 entry, int offset, int value) {
    switch (offset) {
      case 0:
        return entry.withX(value);
      case 1:
        return entry.withY(value);
      case 2:
        return entry.withZ(value);
      case 3:
        return entry.withW(value);
      default:
        throw StateError('Invalid Int32x4 entry offset: $offset/4');
    }
  }

  @override
  Int32x4 addValueFromEntry(Int32x4 entry, int offset, int value) {
    switch (offset) {
      case 0:
        return entry.withX(entry.x + value);
      case 1:
        return entry.withY(entry.y + value);
      case 2:
        return entry.withZ(entry.z + value);
      case 3:
        return entry.withW(entry.w + value);
      default:
        throw StateError('Invalid Float32x4 entry offset: $offset/4');
    }
  }

  @override
  Int32x4 createEntry1(int v0) => Int32x4(v0, 0, 0, 0);

  @override
  Int32x4 createEntry2(int v0, int v1) => Int32x4(v0, v1, 0, 0);

  @override
  Int32x4 createEntry3(int v0, int v1, int v2) => Int32x4(v0, v1, v2, 0);

  @override
  Int32x4 createEntry4(int v0, int v1, int v2, int v3) =>
      Int32x4(v0, v1, v2, v3);

  @override
  Int32x4 createEntryFrom(Int32x4 other, [int? v0, int? v1, int? v2, int? v3]) {
    return Int32x4(v0 ?? other.x, v1 ?? other.y, v2 ?? other.z, v3 ?? other.w);
  }

  @override
  Int32x4 createEntryFullOf(int v) => Int32x4(v, v, v, v);

  static final Int32x4 _entryEmpty = Int32x4(0, 0, 0, 0);

  @override
  Int32x4 get entryEmpty => _entryEmpty;

  static final Random _random = Random();

  @override
  int createRandomValue(int scale, [Random? rand]) {
    rand ??= _random;
    return rand.nextInt(scale * 2) - scale;
  }

  @override
  Int32x4 createRandomEntry(int scale, [Random? rand]) {
    return Int32x4(
        createRandomValue(scale, rand),
        createRandomValue(scale, rand),
        createRandomValue(scale, rand),
        createRandomValue(scale, rand));
  }

  @override
  Int32x4 entryOperationSum(Int32x4 entry1, Int32x4 entry2) {
    return entry1 + entry2;
  }

  @override
  Int32x4 entryOperationSubtract(Int32x4 entry1, Int32x4 entry2) {
    return entry1 - entry2;
  }

  @override
  Int32x4 entryOperationMultiply(Int32x4 entry1, Int32x4 entry2) {
    return entry1 * entry2;
  }

  @override
  Int32x4 entryOperationDivide(Int32x4 entry1, Int32x4 entry2) {
    return entry1 ~/ entry2;
  }

  @override
  int entryOperationSumLane(Int32x4 entry) {
    return entry.sumLane;
  }

  @override
  int entryOperationSumLanePartial(Int32x4 entry, int size) {
    return entry.sumLanePartial(size);
  }

  @override
  int entryOperationSumSquaresLane(Int32x4 entry) {
    return entry.sumSquaresLane;
  }

  @override
  int entryOperationSumSquaresLanePartial(Int32x4 entry, int size) {
    return entry.sumSquaresLanePartial(size);
  }

  @override
  void multiplyTo(SignalInt32x4 other, SignalInt32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];
    }
  }

  @override
  SignalInt32x4 multiply(SignalInt32x4 other) {
    var destiny = SignalInt32x4(capacity);
    multiplyTo(other, destiny);
    return destiny;
  }

  @override
  void subtractTo(SignalInt32x4 other, SignalInt32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];
    }
  }

  @override
  SignalInt32x4 subtract(SignalInt32x4 other) {
    var destiny = SignalInt32x4(capacity);
    subtractTo(other, destiny);
    return destiny;
  }

  @override
  void multiplyAllEntriesTo(Int32x4 entry, SignalInt32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;
    }
  }

  @override
  void subtractAllEntriesTo(Int32x4 entry, SignalInt32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;
    }
  }

  @override
  void multiplyAllEntriesAddingTo(Int32x4 entry, SignalInt32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;
    }
  }

  @override
  SignalInt32x4 multiplyEntries(Int32x4 entry) {
    var destiny = SignalInt32x4(capacity);
    multiplyAllEntriesTo(entry, destiny);
    return destiny;
  }

  @override
  Int32x4 normalizeEntry(Int32x4 entry, Scale<int> scale) => createEntry4(
      scale.normalize(entry.x),
      scale.normalize(entry.y),
      scale.normalize(entry.z),
      scale.normalize(entry.w));

  static final ListEquality<Int32x4> _entriesEquality =
      ListEquality<Int32x4>(Int32x4Equality());

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SignalInt32x4 &&
          runtimeType == other.runtimeType &&
          _size == other._size &&
          _entriesEquality.equals(_entries, other._entries);

  @override
  int get hashCode => _entries.hashCode ^ _size.hashCode;
}

class SignalFloat32x4Mod4 extends SignalFloat32x4 {
  static final SignalFloat32x4Mod4 EMPTY = SignalFloat32x4Mod4(0);

  SignalFloat32x4Mod4(int size)
      : super._(Float32x4List(calcEntriesCapacity(size)), size);

  SignalFloat32x4Mod4._(Float32x4List entries, int size)
      : super._(entries, size);

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 4);

  factory SignalFloat32x4Mod4.from(List<double> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalFloat32x4Mod4.fromEntries(List<Float32x4> entries, int size) =>
      SignalFloat32x4Mod4._(Float32x4List.fromList(entries), size);

  @override
  String get format => 'Float32x4Mod4';

  @override
  SignalFloat32x4Mod4 createInstance(int size) => SignalFloat32x4Mod4(size);

  @override
  SignalFloat32x4Mod4 createRandomInstance(int size, double randomScale,
      [Random? rand]) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale, rand);
    return SignalFloat32x4Mod4.fromEntries(entries, size);
  }

  @override
  SignalFloat32x4Mod4 createInstanceWithEntries(
      int size, List<Float32x4> entries) {
    ensureEntriesLengthMod(entries);
    return SignalFloat32x4Mod4._(Float32x4List.fromList(entries), size);
  }

  @override
  void ensureEntriesLengthMod(List<Float32x4> entries) {
    while (entries.length % 4 != 0) {
      entries.add(entryEmpty);
    }
  }

  @override
  SignalFloat32x4Mod4 createInstanceWithValues(List<double> values) =>
      createInstanceWithEntries(values.length, valuesToEntries(values));

  static final int ENTRY_BLOCK_SIZE = 4;

  @override
  int get entryBlockSize => ENTRY_BLOCK_SIZE;

  @override
  void multiplyTo(SignalFloat32x4 other, SignalFloat32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];

      --i;
      entriesDst[i] = _entries[i] * entries2[i];
    }
  }

  @override
  void subtractTo(SignalFloat32x4 other, SignalFloat32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];

      --i;
      entriesDst[i] = _entries[i] - entries2[i];
    }
  }

  @override
  void multiplyAllEntriesTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;

      --i;
      entriesDst[i] = _entries[i] * entry;
    }
  }

  @override
  void subtractAllEntriesTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;

      --i;
      entriesDst[i] = _entries[i] - entry;
    }
  }

  @override
  void multiplyAllEntriesAddingTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;

      --i;
      entriesDst[i] += _entries[i] * entry;
    }
  }
}

class SignalFloat32x4 extends Signal<double, Float32x4, SignalFloat32x4> {
  static final SignalFloat32x4 EMPTY = SignalFloat32x4(0);

  final Float32x4List _entries;
  final int _entriesLength;

  final int _size;

  SignalFloat32x4(int size)
      : _size = size,
        _entries = Float32x4List(calcEntriesCapacity(size)),
        _entriesLength = calcEntriesCapacity(size) {
    assert(_entries.length == _entriesLength);
  }

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 1);

  SignalFloat32x4._(this._entries, this._size)
      : _entriesLength = _entries.length {
    assert(_entries.length == _entriesLength);
  }

  factory SignalFloat32x4.from(List<double> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalFloat32x4.fromEntries(List<Float32x4> entries, int size) =>
      SignalFloat32x4._(Float32x4List.fromList(entries), size);

  @override
  String get format => 'Float32x4';

  @override
  int calcEntriesCapacityForSize(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 1);

  @override
  double toN(num n) => n.toDouble();

  @override
  String nToString(double n) => formatDecimal(n, precision: 4) ?? '$n';

  static final int ENTRY_BLOCK_SIZE = 4;

  @override
  int get entryBlockSize => ENTRY_BLOCK_SIZE;

  @override
  int get length => _size;

  @override
  int get capacity => _entriesLength * 4;

  @override
  int get entriesLength => _entriesLength;

  @override
  List<Float32x4> get entries => _entries.toList();

  @override
  SignalFloat32x4 copy() =>
      SignalFloat32x4._(Float32x4List.fromList(_entries), _size);

  @override
  List<double> get values {
    final lastEntryIndex = entriesLength - 1;
    var entryIndex = 0;
    var entry = _entries.first;
    var entryCursor = -1;
    var valueCursor = -1;

    var list = List.generate(length, (i) {
      assert(++valueCursor == i);

      switch (++entryCursor) {
        case 0:
          {
            return entry.x;
          }
        case 1:
          {
            return entry.y;
          }
        case 2:
          {
            return entry.z;
          }
        case 3:
          {
            var w = entry.w;
            if (entryIndex < lastEntryIndex) {
              entry = _entries[++entryIndex];
              entryCursor = -1;
            }
            return w;
          }
        default:
          throw StateError('Invalid entryCursor: $entryCursor');
      }
    });

    return list;
  }

  @override
  List<double> get valuesAsDouble => values;

  @override
  double get zero => 0.0;

  @override
  double get one => 1.0;

  @override
  SignalFloat32x4 createInstance(int size) => SignalFloat32x4(size);

  @override
  SignalFloat32x4 createRandomInstance(int size, double randomScale,
      [Random? rand]) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale, rand);
    return SignalFloat32x4.fromEntries(entries, size);
  }

  @override
  SignalFloat32x4 createInstanceWithEntries(int size, List<Float32x4> entries) {
    ensureEntriesLengthMod(entries);
    return SignalFloat32x4._(Float32x4List.fromList(entries), size);
  }

  @override
  void ensureEntriesLengthMod(List<Float32x4> entries) {}

  @override
  Float32x4 getEntry(int index) => _entries[index];

  @override
  void setEntry(int index, Float32x4 entry) => _entries[index] = entry;

  @override
  void addToEntry(int index, Float32x4 entry) {
    _entries[index] += entry;
  }

  @override
  void subtractToEntry(int index, Float32x4 entry) {
    _entries[index] -= entry;
  }

  @override
  Float32x4 getEntryFilteredX4(
      int index, Float32x4 Function(Float32x4 n) filter) {
    return filter(_entries[index]);
  }

  @override
  Float32x4 getEntryFiltered(int index, double Function(double n) filter) {
    var entry = _entries[index];
    return Float32x4(
      filter(entry.x),
      filter(entry.y),
      filter(entry.z),
      filter(entry.w),
    );
  }

  @override
  double getValueFromEntry(Float32x4 entry, int offset) {
    switch (offset) {
      case 0:
        return entry.x;
      case 1:
        return entry.y;
      case 2:
        return entry.z;
      case 3:
        return entry.w;
      default:
        throw StateError('Invalid Float32x4 entry offset: $offset/4');
    }
  }

  @override
  Float32x4 setValueFromEntry(Float32x4 entry, int offset, double value) {
    switch (offset) {
      case 0:
        return entry.withX(value);
      case 1:
        return entry.withY(value);
      case 2:
        return entry.withZ(value);
      case 3:
        return entry.withW(value);
      default:
        throw StateError('Invalid Float32x4 entry offset: $offset/4');
    }
  }

  @override
  Float32x4 addValueFromEntry(Float32x4 entry, int offset, double value) {
    switch (offset) {
      case 0:
        return entry.withX(entry.x + value);
      case 1:
        return entry.withY(entry.y + value);
      case 2:
        return entry.withZ(entry.z + value);
      case 3:
        return entry.withW(entry.w + value);
      default:
        throw StateError('Invalid Float32x4 entry offset: $offset/4');
    }
  }

  @override
  Float32x4 createEntry1(double v0) => Float32x4(v0, 0, 0, 0);

  @override
  Float32x4 createEntry2(double v0, double v1) => Float32x4(v0, v1, 0, 0);

  @override
  Float32x4 createEntry3(double v0, double v1, double v2) =>
      Float32x4(v0, v1, v2, 0);

  @override
  Float32x4 createEntry4(double v0, double v1, double v2, double v3) =>
      Float32x4(v0, v1, v2, v3);

  @override
  Float32x4 createEntryFrom(Float32x4 other,
      [double? v0, double? v1, double? v2, double? v3]) {
    return Float32x4(
        v0 ?? other.x, v1 ?? other.y, v2 ?? other.z, v3 ?? other.w);
  }

  @override
  Float32x4 createEntryFullOf(double v) => Float32x4.splat(v);

  static final Float32x4 _entryEmpty = Float32x4(0, 0, 0, 0);

  @override
  Float32x4 get entryEmpty => _entryEmpty;

  static final Random _random = Random();

  @override
  double createRandomValue(double scale, [Random? rand]) {
    rand ??= _random;
    return (rand.nextDouble() * (scale * 2)) - scale;
  }

  double _createRandomValue(double scale, Random rand) {
    return (rand.nextDouble() * (scale * 2)) - scale;
  }

  @override
  Float32x4 createRandomEntry(double scale, [Random? rand]) {
    rand ??= _random;

    return Float32x4(
        _createRandomValue(scale, rand),
        _createRandomValue(scale, rand),
        _createRandomValue(scale, rand),
        _createRandomValue(scale, rand));
  }

  @override
  Float32x4 entryOperationSum(Float32x4 entry1, Float32x4 entry2) {
    return entry1 + entry2;
  }

  @override
  Float32x4 entryOperationSubtract(Float32x4 entry1, Float32x4 entry2) {
    return entry1 - entry2;
  }

  @override
  Float32x4 entryOperationMultiply(Float32x4 entry1, Float32x4 entry2) {
    return entry1 * entry2;
  }

  @override
  Float32x4 entryOperationDivide(Float32x4 entry1, Float32x4 entry2) {
    return entry1 / entry2;
  }

  @override
  double entryOperationSumLane(Float32x4 entry) {
    return entry.sumLane;
  }

  @override
  double entryOperationSumLanePartial(Float32x4 entry, int size) {
    return entry.sumLanePartial(size);
  }

  @override
  double entryOperationSumSquaresLane(Float32x4 entry) {
    return entry.sumSquaresLane;
  }

  @override
  double entryOperationSumSquaresLanePartial(Float32x4 entry, int size) {
    return entry.sumSquaresLanePartial(size);
  }

  @override
  void multiplyTo(SignalFloat32x4 other, SignalFloat32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entries2[i];
    }
  }

  @override
  SignalFloat32x4 multiply(SignalFloat32x4 other) {
    var destiny = SignalFloat32x4(capacity);
    multiplyTo(other, destiny);
    return destiny;
  }

  @override
  void subtractTo(SignalFloat32x4 other, SignalFloat32x4 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entries2[i];
    }
  }

  @override
  SignalFloat32x4 subtract(SignalFloat32x4 other) {
    var destiny = SignalFloat32x4(capacity);
    subtractTo(other, destiny);
    return destiny;
  }

  @override
  void multiplyAllEntriesTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entry;
    }
  }

  @override
  void subtractAllEntriesTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entry;
    }
  }

  @override
  void multiplyAllEntriesAddingTo(Float32x4 entry, SignalFloat32x4 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] += _entries[i] * entry;
    }
  }

  @override
  SignalFloat32x4 multiplyEntries(Float32x4 entry) {
    var destiny = SignalFloat32x4(capacity);
    multiplyAllEntriesTo(entry, destiny);
    return destiny;
  }

  @override
  Float32x4 normalizeEntry(Float32x4 entry, Scale<double> scale) =>
      createEntry4(scale.normalize(entry.x), scale.normalize(entry.y),
          scale.normalize(entry.z), scale.normalize(entry.w));

  static final ListEquality<Float32x4> _entriesEquality =
      ListEquality<Float32x4>(Float32x4Equality());

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SignalFloat32x4 &&
          runtimeType == other.runtimeType &&
          _size == other._size &&
          _entriesEquality.equals(_entries, other._entries);

  @override
  int get hashCode => _entries.hashCode ^ _size.hashCode;
}

abstract class Signal<N extends num, E, T extends Signal<N, E, T>>
    extends ListBase<N> {
  static int calcNeededBlocksChunks(int size, int blockSize, int chunks) {
    return calcNeededBlocks(calcNeededBlocks(size, blockSize), chunks) * chunks;
  }

  static int calcNeededBlocks(int size, int blockSize) {
    var blocks = size ~/ blockSize;
    var capacity = blocks * blockSize;
    if (capacity < size) blocks++;
    return blocks;
  }

  static Signal<N, E, T>
      fromFormat<N extends num, E, T extends Signal<N, E, T>>(
    String format, {
    int? size,
    List<N>? values,
    List<E>? entries,
  }) {
    switch (format) {
      case 'Float32x4':
        {
          if (values != null) {
            return SignalFloat32x4.from(values.asDoubles()) as Signal<N, E, T>;
          } else if (entries != null) {
            return SignalFloat32x4.fromEntries(
                entries as List<Float32x4>, size!) as Signal<N, E, T>;
          } else {
            return SignalFloat32x4(size!) as Signal<N, E, T>;
          }
        }
      case 'Int32x4':
        {
          if (values != null) {
            return SignalInt32x4.from(values.asInts()) as Signal<N, E, T>;
          } else if (entries != null) {
            return SignalInt32x4.fromEntries(entries as List<Int32x4>, size!)
                as Signal<N, E, T>;
          } else {
            return SignalInt32x4(size!) as Signal<N, E, T>;
          }
        }
      case 'Float32x4Mod4':
        {
          if (values != null) {
            return SignalFloat32x4Mod4.from(values.asDoubles())
                as Signal<N, E, T>;
          } else if (entries != null) {
            return SignalFloat32x4Mod4.fromEntries(
                entries as List<Float32x4>, size!) as Signal<N, E, T>;
          } else {
            return SignalFloat32x4Mod4(size!) as Signal<N, E, T>;
          }
        }
      default:
        throw StateError('Unknown format: $format');
    }
  }

  String get format;

  @override
  N operator [](int valueIndex) => getValue(valueIndex);

  @override
  void operator []=(int valueIndex, N value) => setValue(valueIndex, value);

  @override
  void add(N value) => throw UnsupportedError('Fixed-length list: $length');

  int calcEntriesCapacityForSize(int size);

  /// Number `0` as [N].
  N get zero;

  /// Number `1` as [N].
  N get one;

  /// Converts num [n] to [N].
  N toN(num n);

  /// Converts [n] to [String].
  String nToString(N n);

  /// The block size of an entry.
  int get entryBlockSize;

  @override
  set length(int newLength) =>
      throw UnsupportedError('Fixed-length list: $length');

  @override

  /// Returns [values] length.
  int get length;

  /// returns the values capacity. Can be bigger than [length]
  /// and the value should be "[entryBlockSize] * [entriesLength]".
  int get capacity;

  /// Returns [entries] length.
  int get entriesLength;

  /// Returns all entries.
  List<E> get entries;

  /// Returns the length of the last entry.
  int get lastEntryLength {
    var length = this.length;
    return length == 0 ? 0 : length - (capacity - entryBlockSize);
  }

  /// Returns entry at [index].
  E getEntry(int index);

  /// Set entry at [index] with [entry].
  void setEntry(int index, E entry);

  /// Adds [entry] to entry at [index].
  void addToEntry(int index, E entry);

  void subtractToEntry(int index, E entry);

  E getEntryFilteredX4(int index, E Function(E entry) filter);

  E setEntryFilteredX4(int index, E Function(E entry) filter) {
    var entry = getEntryFilteredX4(index, filter);
    setEntry(index, entry);
    return entry;
  }

  /// Returns entry at [index] with values filtered by [filter].
  E getEntryFiltered(int index, N Function(N n) filter);

  /// Sets entry at [index] with values filtered by [filter].
  E setEntryFiltered(int index, N Function(N n) filter) {
    var entry = getEntryFiltered(index, filter);
    setEntry(index, entry);
    return entry;
  }

  /// Sets entry at [index] with [value].
  E setEntryWithValue(int index, N value) {
    var entry = createEntryFullOf(value);
    setEntry(index, entry);
    return entry;
  }

  /// Sets entry at [index] with a random value.
  E setEntryWithRandomValues(int index, N scale, [Random? rand]) {
    var entry = createRandomEntry(scale, rand);
    setEntry(index, entry);
    return entry;
  }

  /// Sets entry at [index] empty (zeroes).
  void setEntryEmpty(int index) {
    setEntry(index, entryEmpty);
  }

  /// Set all entries empty (zeroes).
  void setAllEntriesEmpty() {
    for (var i = entriesLength - 1; i >= 0; --i) {
      setEntry(i, entryEmpty);
    }
  }

  /// Set all entries with [value].
  void setAllEntriesWithValue(N value) {
    var entry = createEntryFullOf(value);
    for (var i = entriesLength - 1; i >= 0; --i) {
      setEntry(i, entry);
    }
  }

  /// Set all entries with [other] entries.
  void setAllEntriesWith(T other) {
    for (var i = entriesLength - 1; i >= 0; --i) {
      var entry = other.getEntry(i);
      setEntry(i, entry);
    }
  }

  /// Copy instance.
  T copy();

  /// Creates an isntance of same [length] of this.
  T createInstanceOfSameLength() => createInstance(length);

  T createInstanceOfSameLengthFullOfValue(N value) =>
      createInstanceFullOfValue(length, value);

  /// Creates a instance with [size].
  T createInstance(int size);

  /// Creates a random instances.
  T createRandomInstance(int size, N randomScale, [Random? rand]);

  T createInstanceWithEntries(int size, List<E> entries);

  void ensureEntriesLengthMod(List<E> entries);

  /// Creates a [Signal] instance full with [values].
  T createInstanceWithValues(List<N> values) =>
      createInstanceWithEntries(values.length, valuesToEntries(values));

  /// Creates a [Signal] instance full of [value].
  T createInstanceFullOfValue(int size, N value) {
    var o = createInstance(size);
    o.setAllEntriesWithValue(value);
    return o;
  }

  /// Sum operation (SIMD).
  E entryOperationSum(E entry1, E entry2);

  /// Subtract operation (SIMD).
  E entryOperationSubtract(E entry1, E entry2);

  /// Multiply operation (SIMD).
  E entryOperationMultiply(E entry1, E entry2);

  /// Divide operation (SIMD).
  E entryOperationDivide(E entry1, E entry2);

  /// Sum lane operation (SIMD).
  N entryOperationSumLane(E entry);

  /// Sum lane partially (until [size]) operation (SIMD).
  N entryOperationSumLanePartial(E entry, int size);

  /// Sum squares of lane operation (SIMD).
  N entryOperationSumSquaresLane(E entry);

  /// Sum squares of lane partially (until [size]) operation (SIMD).
  N entryOperationSumSquaresLanePartial(E entry, int size);

  /// Multiply all entries with [other] entries and save to [destiny].
  void multiplyTo(T other, T destiny);

  /// Multiply all entries with [other] entries.
  T multiply(T other);

  /// Subtract all entries with [other] and save to [destiny].
  void subtractTo(T other, T destiny);

  /// Subtract all entries with [other] entries.
  T subtract(T other);

  /// Multiply all entries with [entry] and save to [destiny].
  void multiplyAllEntriesTo(E entry, T destiny);

  /// Multiply all entries with [entry] and subtract from [destiny].
  void subtractAllEntriesTo(E entry, T destiny);

  /// Multiply all entries with [entry] and add to [destiny].
  void multiplyAllEntriesAddingTo(E entry, T destiny);

  T multiplyEntries(E entry);

  void multiplyValueTo(N value, T destiny) {
    var entry = createEntryFullOf(value);
    multiplyAllEntriesTo(entry, destiny);
  }

  /// Multiply all values with [value] and add to [destiny].
  void multiplyAllValuesAddingTo(N value, T destiny) {
    var entry = createEntryFullOf(value);
    multiplyAllEntriesAddingTo(entry, destiny);
  }

  int getValueEntryIndex(int valueIndex) => valueIndex ~/ entryBlockSize;

  N getValueFromEntry(E entry, int offset);

  E setValueFromEntry(E entry, int offset, N value);

  E addValueFromEntry(E entry, int offset, N value);

  E createEntry1(N v0) => throw UnsupportedError(_throwMessage_NoOpValues1);

  E createEntry2(N v0, N v1) =>
      throw UnsupportedError(_throwMessage_NoOpValues2);

  E createEntry3(N v0, N v1, N v2) =>
      throw UnsupportedError(_throwMessage_NoOpValues3);

  E createEntry4(N v0, N v1, N v2, N v3) =>
      throw UnsupportedError(_throwMessage_NoOpValues4);

  E createEntryFrom(E other, [N? v0, N? v1, N? v2, N? v3]) =>
      throw UnsupportedError(_throwMessage_NoOpValues1);

  /// Creates an entry with [values].
  E createEntry(List<N> values) {
    switch (values.length) {
      case 1:
        return createEntry1(values[0]);
      case 2:
        return createEntry2(values[0], values[1]);
      case 3:
        return createEntry3(values[0], values[1], values[2]);
      case 4:
        return createEntry4(values[0], values[1], values[2], values[3]);
      default:
        throw StateError('Invalid values size: ${values.length}');
    }
  }

  /// Creates an entry full of value [v].
  E createEntryFullOf(N v);

  /// A constant empty entry (filled with zeroes).
  E get entryEmpty;

  /// Creates a random value.
  N createRandomValue(N scale, [Random? rand]);

  /// Creates a random entry.
  E createRandomEntry(N scale, [Random? rand]);

  /// Creates a list of random entries.
  List<E> createRandomEntries(int size, N randomScale, [Random? rand]) =>
      List.generate(size, (i) => createRandomEntry(randomScale, rand));

  String get _throwMessage_NoOpValues1 =>
      'No operation with 1 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues2 =>
      'No operation with 2 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues3 =>
      'No operation with 3 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues4 =>
      'No operation with 4 value! Entry block size: $entryBlockSize';

  /// Returns a value at [valueIndex].
  N getValue(int valueIndex) {
    var entryIndex = getValueEntryIndex(valueIndex);
    var offset = valueIndex - (entryIndex * entryBlockSize);
    var entry = getEntry(entryIndex);
    return getValueFromEntry(entry, offset);
  }

  /// Sets a value at [valueIndex] with [newValue].
  void setValue(int valueIndex, N newValue) {
    var entryIndex = getValueEntryIndex(valueIndex);
    var offset = valueIndex - (entryIndex * entryBlockSize);
    var entry = getEntry(entryIndex);
    var entry2 = setValueFromEntry(entry, offset, newValue);
    setEntry(entryIndex, entry2);
  }

  /// Adds [value] at [valueIndex].
  void addToValue(int valueIndex, N value) {
    var entryIndex = getValueEntryIndex(valueIndex);
    var offset = valueIndex - (entryIndex * entryBlockSize);
    var entry = getEntry(entryIndex);
    var entry2 = addValueFromEntry(entry, offset, value);
    setEntry(entryIndex, entry2);
  }

  /// Sets entry at [entryIndex] and value 1 to [v0].
  void setEntryValues1(int entryIndex, N v0) =>
      setEntry(entryIndex, createEntry1(v0));

  /// Sets entry at [entryIndex] and values 1 and 2 to [v0] and [v1].
  void setEntryValues2(int entryIndex, N v0, N v1) =>
      setEntry(entryIndex, createEntry2(v0, v1));

  /// Sets entry at [entryIndex] and values 1, 2 and 3 to [v0], [v1] and [v2].
  void setEntryValues3(int entryIndex, N v0, N v1, N v2) =>
      setEntry(entryIndex, createEntry3(v0, v1, v2));

  /// Sets entry at [entryIndex] and values 1, 2, 3 and 4 to [v0], [v1], [v2] and [v3].
  void setEntryValues4(int entryIndex, N v0, N v1, N v2, N v3) =>
      setEntry(entryIndex, createEntry4(v0, v1, v2, v3));

  /// Set extra values to [zero]. Calls [setExtraValues] with [zero].
  void setExtraValuesToZero() => setExtraValues(zero);

  /// Set extra values to [one]. Calls [setExtraValues] with [one].
  void setExtraValuesToOne() => setExtraValues(one);

  /// Set the extra values (values over length) with [value].
  ///
  /// Since the values are stored in entries blocks (SIMD pattern),
  /// depending of [length], some extra values beyond [length] will exists.
  void setExtraValues(N value) {
    var length = this.length;
    var capacity = this.capacity;
    var extraSize = capacity - length;

    if (extraSize == 0) return;

    if (entryBlockSize == 4) {
      var lastEntryIndex = entriesLength - 1;
      var lastEntry = getEntry(lastEntryIndex);

      E entry;

      switch (extraSize) {
        case 1:
          {
            entry = createEntryFrom(lastEntry, null, null, null, value);
            break;
          }
        case 2:
          {
            entry = createEntryFrom(lastEntry, null, null, value, value);
            break;
          }
        case 3:
          {
            entry = createEntryFrom(lastEntry, null, value, value, value);
            break;
          }
        default:
          throw StateError('Unreachable state: $extraSize / $entryBlockSize');
      }

      setEntry(lastEntryIndex, entry);
    } else {
      for (var i = length; i < capacity; ++i) {
        setValue(i, value);
      }
    }
  }

  List<E> getEntries([int? length]) {
    length ??= entriesLength;
    if (length <= 0) return <E>[];
    if (length > entriesLength) length = entriesLength;
    return List.generate(length, (i) => getEntry(i));
  }

  /// Returns a [DataStatistics] of [values].
  DataStatistics get statistics => DataStatistics.compute(values);

  /// Returns the values as a [List<N>].
  List<N> get values => List.generate(length, (i) => getValue(i));

  /// Returns the [values] as a [List<double>].
  List<double> get valuesAsDouble =>
      List.generate(length, (i) => getValue(i).toDouble());

  /// Returns the [values] as a [List<String>].
  List<String> get valuesAsString =>
      List.generate(length, (i) => nToString(getValue(i)));

  /// Get [values] with [length].
  List<N> getValues([int? length]) {
    length ??= this.length;
    if (length <= 0) return <N>[];
    if (length > this.length) length = this.length;
    return List.generate(length, (i) => getValue(i));
  }

  /// Converts [values] to a List of entries [E].
  List<E> valuesToEntries(List<N> values) {
    var size = values.length;

    var entryBlockSize = this.entryBlockSize;
    var sizeMod = (size ~/ entryBlockSize) * entryBlockSize;

    var entries = <E>[];

    var i = 0;
    if (entryBlockSize == 4) {
      for (; i < sizeMod;) {
        var v0 = values[i++];
        var v1 = values[i++];
        var v2 = values[i++];
        var v3 = values[i++];
        var entry = createEntry4(v0, v1, v2, v3);
        entries.add(entry);
      }
    } else if (entryBlockSize == 3) {
      for (; i < sizeMod;) {
        var v0 = values[i++];
        var v1 = values[i++];
        var v2 = values[i++];
        var entry = createEntry3(v0, v1, v2);
        entries.add(entry);
      }
    } else if (entryBlockSize == 2) {
      for (; i < sizeMod;) {
        var v0 = values[i++];
        var v1 = values[i++];
        var entry = createEntry2(v0, v1);
        entries.add(entry);
      }
    } else if (entryBlockSize == 1) {
      for (; i < sizeMod;) {
        var v0 = values[i++];
        var entry = createEntry1(v0);
        entries.add(entry);
      }
    }

    if (i < size) {
      var diff = size - sizeMod;

      E tailEntry;
      switch (diff) {
        case 3:
          {
            var v0 = values[i++];
            var v1 = values[i++];
            var v2 = values[i++];
            tailEntry = createEntry3(v0, v1, v2);
            break;
          }
        case 2:
          {
            var v0 = values[i++];
            var v1 = values[i++];
            tailEntry = createEntry2(v0, v1);
            break;
          }
        case 1:
          {
            var v0 = values[i++];
            tailEntry = createEntry1(v0);
            break;
          }
        default:
          throw StateError(
              'Invalid state: $size - $sizeMod = $diff (entryBlockSize: $entryBlockSize)');
      }

      entries.add(tailEntry);
    }

    return entries;
  }

  void set(T other, [int? entriesLength]) {
    entriesLength ??= this.entriesLength;

    for (var i = entriesLength - 1; i >= 0; --i) {
      var entry = other.getEntry(i);
      setEntry(i, entry);
    }
  }

  /// Normalizes an entry [E] with [scale].
  E normalizeEntry(E entry, Scale<N> scale);

  /// Creates a [Signal] instance normalized with [scale].
  T normalize(Scale<N> scale) {
    var entries2 = entries.map((e) => normalizeEntry(e, scale)).toList();
    return createInstanceWithEntries(length, entries2);
  }

  /// Computes the sum of squares mean of [values].
  double computeSumSquaresMean() {
    return computeSumSquares() / length;
  }

  /// Computes the sum of sqaues of [values].
  N computeSumSquares() {
    var length = this.length;
    if (length == 0) return toN(zero);

    var entriesLength = this.entriesLength;

    num total;
    {
      var lastEntry = getEntry(entriesLength - 1);
      total = entryOperationSumSquaresLanePartial(lastEntry, lastEntryLength);
    }

    for (var i = entriesLength - 2; i >= 0; --i) {
      var entry = getEntry(i);
      var sumSquares = entryOperationSumSquaresLane(entry);
      total += sumSquares;
    }

    return toN(total);
  }

  /// Returns the differences of this instances [values] to [otherValues].
  List<N> diff(List<N> otherValues) => List.generate(
      otherValues.length, (i) => toN(getValue(i) - otherValues[i]));

  /// Returns the absolute differences of this instances [values] to [otherValues].
  List<N> diffAbs(List<N> values) =>
      List.generate(values.length, (i) => toN((getValue(i) - values[i]).abs()));

  /// Same as [diff].
  List<N> errors(List<N> output) => diff(output);

  /// Same as [diffAbs].
  List<N> errorsAbs(List<N> output) => diffAbs(output);

  /// Returns the mean of [errors].
  double errorGlobalMean(List<N> output) => errors(output).mean;

  /// Returns the square mean of [errors].
  double errorGlobalSquareMean(List<N> output) => errors(output).squaresMean;

  /// Returns the square mean root of [errors].
  double errorGlobalSquareMeanRoot(List<N> output) =>
      errors(output).squaresMean.squareRoot;

  @override
  String toString(
      {int maxElements = 10, bool entries = false, bool infos = false}) {
    if (infos) {
      return entries
          ? toStringWithEntries(maxElements)
          : toStringWithValues(maxElements);
    } else {
      return entries
          ? toStringEntries(maxElements)
          : toStringValues(maxElements);
    }
  }

  String toStringWithValues([int maxValuesToString = 10]) {
    return '$runtimeType{length: $length, values: ${toStringValues(maxValuesToString)}';
  }

  String toStringWithEntries([int maxEntriesToString = 4]) {
    return '$runtimeType{length: $length, entries: ${toStringEntries(maxEntriesToString)}';
  }

  String toStringValues([int maxValuesToString = 10]) {
    return length < maxValuesToString
        ? '$valuesAsString'
        : '${getValues(maxValuesToString)}...[#$length]';
  }

  String toStringEntries([int maxEntriesToString = 4]) {
    return entriesLength < maxEntriesToString
        ? '$entries'
        : '${getEntries(maxEntriesToString)}...[#$entriesLength]';
  }
}
