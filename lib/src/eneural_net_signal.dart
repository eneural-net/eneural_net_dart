import 'dart:collection';
import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:swiss_knife/swiss_knife.dart' show formatDecimal;

import 'eneural_net_extension.dart';
import 'eneural_net_scale.dart';

class SignalInt32 extends Signal<int, Int32x4, SignalInt32> {
  static final SignalInt32 EMPTY = SignalInt32(0);

  final Int32x4List _entries;
  final int _entriesLength;

  final int _size;

  SignalInt32(int size)
      : _size = size,
        _entries = Int32x4List(calcEntriesCapacity(size)),
        _entriesLength = calcEntriesCapacity(size);

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 4);

  SignalInt32._(this._entries, this._size) : _entriesLength = _entries.length;

  factory SignalInt32.from(List<int> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalInt32.fromEntries(List<Int32x4> entries, int size) =>
      SignalInt32._(Int32x4List.fromList(entries), size);

  @override
  int calcEntriesCapacityForSize(int size) =>
      Signal.calcNeededBlocksChunks(size, 4, 4);

  @override
  SignalInt32 copy() => SignalInt32._(Int32x4List.fromList(_entries), _size);

  @override
  int get zero => 0;

  @override
  int get one => 1;

  @override
  SignalInt32 createInstance(int size) => SignalInt32(size);

  @override
  SignalInt32 createRandomInstance(int size, int randomScale) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale);
    return SignalInt32.fromEntries(entries, size);
  }

  @override
  SignalInt32 createInstanceWithEntries(int size, List<Int32x4> entries) {
    while (entries.length % 4 != 0) {
      entries.add(entryEmpty);
    }

    return SignalInt32._(Int32x4List.fromList(entries), size);
  }

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
  int get capacity => _entries.length * 4;

  @override
  int get entriesLength => _entriesLength;

  @override
  List<Int32x4> get entries => _entries.toList();

  @override
  Int32x4 getEntry(int index) => _entries[index];

  @override
  void setEntry(int index, Int32x4 entry) => _entries[index] = entry;

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
  Int32x4 createEntry1(int v0) => Int32x4(v0, 0, 0, 0);

  @override
  Int32x4 createEntry2(int v0, int v1) => Int32x4(v0, v1, 0, 0);

  @override
  Int32x4 createEntry3(int v0, int v1, int v2) => Int32x4(v0, v1, v2, 0);

  @override
  Int32x4 createEntry4(int v0, int v1, int v2, int v3) =>
      Int32x4(v0, v1, v2, v3);

  @override
  Int32x4 createEntryFullOf(int v) => Int32x4(v, v, v, v);

  static final Int32x4 _entryEmpty = Int32x4(0, 0, 0, 0);

  @override
  Int32x4 get entryEmpty => _entryEmpty;

  static final Random _random = Random();

  @override
  int createRandomValue(int scale) {
    return _random.nextInt(scale * 2) - scale;
  }

  @override
  Int32x4 createRandomEntry(int scale) {
    return Int32x4(createRandomValue(scale), createRandomValue(scale),
        createRandomValue(scale), createRandomValue(scale));
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
  void multiplyTo(SignalInt32 other, SignalInt32 destiny) {
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
  SignalInt32 multiply(SignalInt32 other) {
    var destiny = SignalInt32(capacity);
    multiplyTo(other, destiny);
    return destiny;
  }

  @override
  void subtractTo(SignalInt32 other, SignalInt32 destiny) {
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
  SignalInt32 subtract(SignalInt32 other) {
    var destiny = SignalInt32(capacity);
    subtractTo(other, destiny);
    return destiny;
  }

  @override
  void multiplyEntryTo(Int32x4 entry, SignalInt32 destiny) {
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
  void subtractEntryTo(Int32x4 entry, SignalInt32 destiny) {
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
  void multiplyEntryAddingTo(Int32x4 entry, SignalInt32 destiny) {
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
  SignalInt32 multiplyEntry(Int32x4 entry) {
    var destiny = SignalInt32(capacity);
    multiplyEntryTo(entry, destiny);
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
      other is SignalInt32 &&
          runtimeType == other.runtimeType &&
          _size == other._size &&
          _entriesEquality.equals(_entries, other._entries);

  @override
  int get hashCode => _entries.hashCode ^ _size.hashCode;
}

class SignalFloat32Mod4 extends SignalFloat32 {
  static final SignalFloat32Mod4 EMPTY = SignalFloat32Mod4(0);

  SignalFloat32Mod4(int size)
      : super._(Float32x4List(calcEntriesCapacity(size)), size);

  SignalFloat32Mod4._(Float32x4List entries, int size) : super._(entries, size);

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 4);

  factory SignalFloat32Mod4.from(List<double> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalFloat32Mod4.fromEntries(List<Float32x4> entries, int size) =>
      SignalFloat32Mod4._(Float32x4List.fromList(entries), size);

  @override
  SignalFloat32Mod4 createInstance(int size) => SignalFloat32Mod4(size);

  @override
  SignalFloat32Mod4 createRandomInstance(int size, int randomScale) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale);
    return SignalFloat32Mod4.fromEntries(entries, size);
  }

  @override
  SignalFloat32Mod4 createInstanceWithEntries(
      int size, List<Float32x4> entries) {
    while (entries.length % 4 != 0) {
      entries.add(entryEmpty);
    }
    return SignalFloat32Mod4._(Float32x4List.fromList(entries), size);
  }

  @override
  SignalFloat32Mod4 createInstanceWithValues(List<double> values) =>
      createInstanceWithEntries(values.length, valuesToEntries(values));

  static final int ENTRY_BLOCK_SIZE = 4;

  @override
  int get entryBlockSize => ENTRY_BLOCK_SIZE;

  @override
  void multiplyTo(SignalFloat32 other, SignalFloat32 destiny) {
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
  void subtractTo(SignalFloat32 other, SignalFloat32 destiny) {
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
  void multiplyEntryTo(Float32x4 entry, SignalFloat32 destiny) {
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
  void subtractEntryTo(Float32x4 entry, SignalFloat32 destiny) {
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
  void multiplyEntryAddingTo(Float32x4 entry, SignalFloat32 destiny) {
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

class SignalFloat32 extends Signal<double, Float32x4, SignalFloat32> {
  static final SignalFloat32 EMPTY = SignalFloat32(0);

  final Float32x4List _entries;
  final int _entriesLength;

  final int _size;

  SignalFloat32(int size)
      : _size = size,
        _entries = Float32x4List(calcEntriesCapacity(size)),
        _entriesLength = calcEntriesCapacity(size);

  static int calcEntriesCapacity(int size) =>
      Signal.calcNeededBlocksChunks(size, ENTRY_BLOCK_SIZE, 1);

  SignalFloat32._(this._entries, this._size) : _entriesLength = _entries.length;

  factory SignalFloat32.from(List<double> values) =>
      EMPTY.createInstanceWithValues(values);

  factory SignalFloat32.fromEntries(List<Float32x4> entries, int size) =>
      SignalFloat32._(Float32x4List.fromList(entries), size);

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
  int get capacity => _entries.length * 4;

  @override
  int get entriesLength => _entriesLength;

  @override
  List<Float32x4> get entries => _entries.toList();

  @override
  SignalFloat32 copy() =>
      SignalFloat32._(Float32x4List.fromList(_entries), _size);

  @override
  double get zero => 0.0;

  @override
  double get one => 1.0;

  @override
  SignalFloat32 createInstance(int size) => SignalFloat32(size);

  @override
  SignalFloat32 createRandomInstance(int size, int randomScale) {
    var capacity = calcEntriesCapacityForSize(size);
    var entries = createRandomEntries(capacity, randomScale);
    return SignalFloat32.fromEntries(entries, size);
  }

  @override
  SignalFloat32 createInstanceWithEntries(int size, List<Float32x4> entries) {
    while (entries.length % 4 != 0) {
      entries.add(entryEmpty);
    }
    return SignalFloat32._(Float32x4List.fromList(entries), size);
  }

  @override
  Float32x4 getEntry(int index) => _entries[index];

  @override
  void setEntry(int index, Float32x4 entry) => _entries[index] = entry;

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
  Float32x4 createEntryFullOf(double v) => Float32x4.splat(v);

  static final Float32x4 _entryEmpty = Float32x4(0, 0, 0, 0);

  @override
  Float32x4 get entryEmpty => _entryEmpty;

  static final Random _random = Random();

  @override
  double createRandomValue(int scale) {
    return (_random.nextDouble() * (scale * 2)) - scale;
  }

  @override
  Float32x4 createRandomEntry(int scale) {
    return Float32x4(createRandomValue(scale), createRandomValue(scale),
        createRandomValue(scale), createRandomValue(scale));
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
  void multiplyTo(SignalFloat32 other, SignalFloat32 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entries2[i];
    }
  }

  @override
  SignalFloat32 multiply(SignalFloat32 other) {
    var destiny = SignalFloat32(capacity);
    multiplyTo(other, destiny);
    return destiny;
  }

  @override
  void subtractTo(SignalFloat32 other, SignalFloat32 destiny) {
    var entries2 = other._entries;
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entries2[i];
    }
  }

  @override
  SignalFloat32 subtract(SignalFloat32 other) {
    var destiny = SignalFloat32(capacity);
    subtractTo(other, destiny);
    return destiny;
  }

  @override
  void multiplyEntryTo(Float32x4 entry, SignalFloat32 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] * entry;
    }
  }

  @override
  void subtractEntryTo(Float32x4 entry, SignalFloat32 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] = _entries[i] - entry;
    }
  }

  @override
  void multiplyEntryAddingTo(Float32x4 entry, SignalFloat32 destiny) {
    var entriesDst = destiny._entries;

    for (var i = _entriesLength; i > 0;) {
      --i;
      entriesDst[i] += _entries[i] * entry;
    }
  }

  @override
  SignalFloat32 multiplyEntry(Float32x4 entry) {
    var destiny = SignalFloat32(capacity);
    multiplyEntryTo(entry, destiny);
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
      other is SignalFloat32 &&
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

  @override
  N operator [](int valueIndex) => getValue(valueIndex);

  @override
  void operator []=(int valueIndex, N value) => setValue(valueIndex, value);

  @override
  void add(N value) => throw UnsupportedError('Fixed-length list: $length');

  int calcEntriesCapacityForSize(int size);

  N get zero;

  N get one;

  N toN(num n);

  String nToString(N n);

  int get entryBlockSize;

  @override
  set length(int newLength) =>
      throw UnsupportedError('Fixed-length list: $length');

  @override
  int get length;

  int get capacity;

  int get entriesLength;

  List<E> get entries;

  E getEntry(int index);

  void setEntry(int index, E entry);

  E getEntryFilteredX4(int index, E Function(E entry) filter);

  E setEntryFilteredX4(int index, E Function(E entry) filter) {
    var entry = getEntryFilteredX4(index, filter);
    setEntry(index, entry);
    return entry;
  }

  E getEntryFiltered(int index, N Function(N n) filter);

  E setEntryFiltered(int index, N Function(N n) filter) {
    var entry = getEntryFiltered(index, filter);
    setEntry(index, entry);
    return entry;
  }

  E setEntryWithValue(int index, N value) {
    var entry = createEntryFullOf(value);
    setEntry(index, entry);
    return entry;
  }

  E setEntryWithRandomValues(int index, int scale) {
    var entry = createRandomEntry(scale);
    setEntry(index, entry);
    return entry;
  }

  void setEntryEmpty(int index) {
    setEntry(index, entryEmpty);
  }

  T copy();

  T createInstance(int size);

  T createRandomInstance(int size, int randomScale);

  T createInstanceWithEntries(int size, List<E> entries);

  T createInstanceWithValues(List<N> values) =>
      createInstanceWithEntries(values.length, valuesToEntries(values));

  E entryOperationSum(E entry1, E entry2);

  E entryOperationSubtract(E entry1, E entry2);

  E entryOperationMultiply(E entry1, E entry2);

  E entryOperationDivide(E entry1, E entry2);

  N entryOperationSumLane(E entry);

  void multiplyTo(T other, T destiny);

  T multiply(T other);

  void subtractTo(T other, T destiny);

  T subtract(T other);

  void multiplyEntryTo(E entry, T destiny);

  void subtractEntryTo(E entry, T destiny);

  void multiplyEntryAddingTo(E entry, T destiny);

  T multiplyEntry(E entry);

  void multiplyValueTo(N value, T destiny) {
    var entry = createEntryFullOf(value);
    multiplyEntryTo(entry, destiny);
  }

  void multiplyValueAddingTo(N value, T destiny) {
    var entry = createEntryFullOf(value);
    multiplyEntryAddingTo(entry, destiny);
  }

  int getValueEntryIndex(int valueIndex) => valueIndex ~/ entryBlockSize;

  N getValueFromEntry(E entry, int offset);

  E setValueFromEntry(E entry, int offset, N value);

  E createEntry1(N v0) => throw UnsupportedError(_throwMessage_NoOpValues1);

  E createEntry2(N v0, N v1) =>
      throw UnsupportedError(_throwMessage_NoOpValues2);

  E createEntry3(N v0, N v1, N v2) =>
      throw UnsupportedError(_throwMessage_NoOpValues3);

  E createEntry4(N v0, N v1, N v2, N v3) =>
      throw UnsupportedError(_throwMessage_NoOpValues4);

  E createEntryFullOf(N v);

  E get entryEmpty;

  N createRandomValue(int scale);

  E createRandomEntry(int scale);

  List<E> createRandomEntries(int size, int randomScale) =>
      List.generate(size, (i) => createRandomEntry(randomScale));

  String get _throwMessage_NoOpValues1 =>
      'No operation with 1 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues2 =>
      'No operation with 2 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues3 =>
      'No operation with 3 value! Entry block size: $entryBlockSize';

  String get _throwMessage_NoOpValues4 =>
      'No operation with 4 value! Entry block size: $entryBlockSize';

  N getValue(int valueIndex) {
    var entryIndex = getValueEntryIndex(valueIndex);
    var offset = valueIndex - (entryIndex * entryBlockSize);
    var entry = getEntry(entryIndex);
    return getValueFromEntry(entry, offset);
  }

  void setValue(int valueIndex, N value) {
    var entryIndex = getValueEntryIndex(valueIndex);
    var offset = valueIndex - (entryIndex * entryBlockSize);
    var entry = getEntry(entryIndex);
    var entry2 = setValueFromEntry(entry, offset, value);
    setEntry(entryIndex, entry2);
  }

  void setEntryValues1(int entryIndex, N v0) =>
      setEntry(entryIndex, createEntry1(v0));

  void setEntryValues2(int entryIndex, N v0, N v1) =>
      setEntry(entryIndex, createEntry2(v0, v1));

  void setEntryValues3(int entryIndex, N v0, N v1, N v2) =>
      setEntry(entryIndex, createEntry3(v0, v1, v2));

  void setEntryValues4(int entryIndex, N v0, N v1, N v2, N v3) =>
      setEntry(entryIndex, createEntry4(v0, v1, v2, v3));

  void setExtraValues(N value) {
    var capacity = this.capacity;
    for (var i = length; i < capacity; ++i) {
      setValue(i, value);
    }
  }

  List<E> getEntries([int? length]) {
    length ??= entriesLength;
    if (length <= 0) return <E>[];
    if (length > entriesLength) length = entriesLength;
    return List.generate(length, (i) => getEntry(i));
  }

  List<N> get values => List.generate(length, (i) => getValue(i));

  List<String> get valuesAsString =>
      List.generate(length, (i) => nToString(getValue(i)));

  List<N> getValues([int? length]) {
    length ??= this.length;
    if (length <= 0) return <N>[];
    if (length > this.length) length = this.length;
    return List.generate(length, (i) => getValue(i));
  }

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

  void set(T other) {
    for (var i = entriesLength - 1; i >= 0; --i) {
      var entry = other.getEntry(i);
      setEntry(i, entry);
    }
  }

  E normalizeEntry(E entry, Scale<N> scale);

  T normalize(Scale<N> scale) {
    var entries2 = entries.map((e) => normalizeEntry(e, scale)).toList();
    return createInstanceWithEntries(length, entries2);
  }

  List<N> diff(List<N> values) =>
      List.generate(values.length, (i) => toN(getValue(i) - values[i]));

  List<N> diffAbs(List<N> values) =>
      List.generate(values.length, (i) => toN((getValue(i) - values[i]).abs()));

  List<N> errors(List<N> output) => diff(output);

  List<N> errorsAbs(List<N> output) => diffAbs(output);

  double errorGlobalMean(List<N> output) => errors(output).mean;

  double errorGlobalSquareMean(List<N> output) => errors(output).square.mean;

  double errorGlobalSquareMeanRoot(List<N> output) =>
      errors(output).square.mean.squareRoot;

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
        ? '$values'
        : '${getValues(maxValuesToString)}...[#$length]';
  }

  String toStringEntries([int maxEntriesToString = 4]) {
    return entriesLength < maxEntriesToString
        ? '$entries'
        : '${getEntries(maxEntriesToString)}...[#$entriesLength]';
  }
}
