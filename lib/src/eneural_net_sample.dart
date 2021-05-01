import 'dart:typed_data';

import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';

class SampleInt32x4 extends Sample<int, Int32x4, SignalInt32x4, Scale<int>> {
  static final SampleInt32x4 DUMMY = SampleInt32x4.normalized(
      SignalInt32x4.from([0]), SignalInt32x4.from([0]), ScaleInt.ZERO_TO_ONE);

  static List<SampleInt32x4> toList(
      List<List<List<int>>> pairs, Scale<int> scale) {
    return pairs.map((p) => SampleInt32x4.from(p[0], p[1], scale)).toList();
  }

  static List<SampleInt32x4> toListFromString(
      List<String> pairs, Scale<int> scale, bool normalized) {
    return pairs
        .map((p) => SampleInt32x4.fromString(p, scale, normalized))
        .toList();
  }

  factory SampleInt32x4.fromString(
      String s, Scale<int> scale, bool normalized) {
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
        ? SampleInt32x4.fromNormalized(input, output, scale)
        : SampleInt32x4.from(input, output, scale);
  }

  SampleInt32x4.normalized(
      SignalInt32x4 input, SignalInt32x4 output, Scale<int> scale)
      : super.normalized(input, output, scale);

  factory SampleInt32x4.fromNormalized(
          List<int> input, List<int> output, Scale<int> scale) =>
      SampleInt32x4.normalized(
          SignalInt32x4.from(input), SignalInt32x4.from(output), scale);

  factory SampleInt32x4(
      SignalInt32x4 input, SignalInt32x4 output, Scale<int> scale) {
    return SampleInt32x4.normalized(DUMMY.normalizeWithScale(input, scale),
        DUMMY.normalizeWithScale(output, scale), scale);
  }

  factory SampleInt32x4.from(
      List<int> input, List<int> output, Scale<int> scale) {
    return SampleInt32x4.normalized(
        DUMMY.normalizeWithScale(SignalInt32x4.from(input), scale),
        DUMMY.normalizeWithScale(SignalInt32x4.from(output), scale),
        scale);
  }
}

/// [ANN] sample based in [Float32x4] data.
class SampleFloat32x4
    extends Sample<double, Float32x4, SignalFloat32x4, Scale<double>> {
  static final SampleFloat32x4 DUMMY = SampleFloat32x4.normalized(
      SignalFloat32x4.from([0]),
      SignalFloat32x4.from([0]),
      ScaleDouble.ZERO_TO_ONE);

  static List<SampleFloat32x4> toList(
      List<List<List<double>>> pairs, Scale<double> scale) {
    return pairs.map((p) => SampleFloat32x4.from(p[0], p[1], scale)).toList();
  }

  static List<SampleFloat32x4> toListFromString(
      List<String> pairs, Scale<double> scale, bool normalized) {
    return pairs
        .map((p) => SampleFloat32x4.fromString(p, scale, normalized))
        .toList();
  }

  factory SampleFloat32x4.fromString(
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
        ? SampleFloat32x4.fromNormalized(input, output, scale)
        : SampleFloat32x4.from(input, output, scale);
  }

  SampleFloat32x4.normalized(
      SignalFloat32x4 input, SignalFloat32x4 output, Scale<double> scale)
      : super.normalized(input, output, scale);

  factory SampleFloat32x4.fromNormalized(
          List<double> input, List<double> output, Scale<double> scale) =>
      SampleFloat32x4(
          SignalFloat32x4.from(input), SignalFloat32x4.from(output), scale);

  factory SampleFloat32x4(
      SignalFloat32x4 input, SignalFloat32x4 output, Scale<double> scale) {
    return SampleFloat32x4.normalized(DUMMY.normalizeWithScale(input, scale),
        DUMMY.normalizeWithScale(output, scale), scale);
  }

  factory SampleFloat32x4.from(
      List<double> input, List<double> output, Scale<double> scale) {
    return SampleFloat32x4.normalized(
        DUMMY.normalizeWithScale(SignalFloat32x4.from(input), scale),
        DUMMY.normalizeWithScale(SignalFloat32x4.from(output), scale),
        scale);
  }
}

/// Base class for [ANN] samples.
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
