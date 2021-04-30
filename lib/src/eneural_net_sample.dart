import 'dart:typed_data';

import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';

class SampleInt32 extends Sample<int, Int32x4, SignalInt32, Scale<int>> {
  static final SampleInt32 DUMMY = SampleInt32.normalized(
      SignalInt32.from([0]), SignalInt32.from([0]), ScaleInt.ZERO_TO_ONE);

  static List<SampleInt32> toList(
      List<List<List<int>>> pairs, Scale<int> scale) {
    return pairs.map((p) => SampleInt32.from(p[0], p[1], scale)).toList();
  }

  static List<SampleInt32> toListFromString(
      List<String> pairs, Scale<int> scale, bool normalized) {
    return pairs
        .map((p) => SampleInt32.fromString(p, scale, normalized))
        .toList();
  }

  factory SampleInt32.fromString(String s, Scale<int> scale, bool normalized) {
    var inOut = s.split(Sample.REGEXP_IN_OUT_DELIMITER);
    var input = inOut[0]
        .split(Sample.REGEXP_VALUE_DELIMITER)
        .map((e) => int.parse(e.trim()))
        .toList();
    var output = inOut[1]
        .split(Sample.REGEXP_VALUE_DELIMITER)
        .map((e) => int.parse(e.trim()))
        .toList();
    return normalized
        ? SampleInt32.fromNormalized(input, output, scale)
        : SampleInt32.from(input, output, scale);
  }

  SampleInt32.normalized(
      SignalInt32 input, SignalInt32 output, Scale<int> scale)
      : super.normalized(input, output, scale);

  factory SampleInt32.fromNormalized(
          List<int> input, List<int> output, Scale<int> scale) =>
      SampleInt32.normalized(
          SignalInt32.from(input), SignalInt32.from(output), scale);

  factory SampleInt32(SignalInt32 input, SignalInt32 output, Scale<int> scale) {
    return SampleInt32.normalized(DUMMY.normalizeWithScale(input, scale),
        DUMMY.normalizeWithScale(output, scale), scale);
  }

  factory SampleInt32.from(
      List<int> input, List<int> output, Scale<int> scale) {
    return SampleInt32.normalized(
        DUMMY.normalizeWithScale(SignalInt32.from(input), scale),
        DUMMY.normalizeWithScale(SignalInt32.from(output), scale),
        scale);
  }
}

class SampleFloat32
    extends Sample<double, Float32x4, SignalFloat32, Scale<double>> {
  static final SampleFloat32 DUMMY = SampleFloat32.normalized(
      SignalFloat32.from([0]),
      SignalFloat32.from([0]),
      ScaleDouble.ZERO_TO_ONE);

  static List<SampleFloat32> toList(
      List<List<List<double>>> pairs, Scale<double> scale) {
    return pairs.map((p) => SampleFloat32.from(p[0], p[1], scale)).toList();
  }

  static List<SampleFloat32> toListFromString(
      List<String> pairs, Scale<double> scale, bool normalized) {
    return pairs
        .map((p) => SampleFloat32.fromString(p, scale, normalized))
        .toList();
  }

  factory SampleFloat32.fromString(
      String s, Scale<double> scale, bool normalized) {
    var inOut = s.split(Sample.REGEXP_IN_OUT_DELIMITER);
    var input = inOut[0]
        .split(Sample.REGEXP_VALUE_DELIMITER)
        .map((e) => double.parse(e.trim()))
        .toList();
    var output = inOut[1]
        .split(Sample.REGEXP_VALUE_DELIMITER)
        .map((e) => double.parse(e.trim()))
        .toList();
    return normalized
        ? SampleFloat32.fromNormalized(input, output, scale)
        : SampleFloat32.from(input, output, scale);
  }

  SampleFloat32.normalized(
      SignalFloat32 input, SignalFloat32 output, Scale<double> scale)
      : super.normalized(input, output, scale);

  factory SampleFloat32.fromNormalized(
          List<double> input, List<double> output, Scale<double> scale) =>
      SampleFloat32(
          SignalFloat32.from(input), SignalFloat32.from(output), scale);

  factory SampleFloat32(
      SignalFloat32 input, SignalFloat32 output, Scale<double> scale) {
    return SampleFloat32.normalized(DUMMY.normalizeWithScale(input, scale),
        DUMMY.normalizeWithScale(output, scale), scale);
  }

  factory SampleFloat32.from(
      List<double> input, List<double> output, Scale<double> scale) {
    return SampleFloat32.normalized(
        DUMMY.normalizeWithScale(SignalFloat32.from(input), scale),
        DUMMY.normalizeWithScale(SignalFloat32.from(output), scale),
        scale);
  }
}

abstract class Sample<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> {
  static final RegExp REGEXP_IN_OUT_DELIMITER = RegExp(r'\s*=\s*');

  static final RegExp REGEXP_VALUE_DELIMITER = RegExp(r'\s*[,;]\s*');

  final T input;

  final T output;

  final S scale;

  Sample.normalized(this.input, this.output, this.scale);

  T normalize(T signal) => signal.normalize(scale);

  T normalizeWithScale(T signal, S scale) => signal.normalize(scale);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Sample &&
          runtimeType == other.runtimeType &&
          scale == other.scale &&
          input == other.input &&
          output == other.output;

  @override
  int get hashCode => input.hashCode ^ output.hashCode ^ scale.hashCode;

  @override
  String toString() {
    return '$runtimeType{${input.valuesAsString} -> ${output.valuesAsString} ; $scale}';
  }
}
