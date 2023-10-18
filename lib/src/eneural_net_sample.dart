import 'dart:math' as math;
import 'dart:typed_data';

//import 'package:eneural_net/eneural_net.dart';

import 'eneural_net_extension.dart';
import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';
import 'eneural_net_tools.dart';

class SampleInt32x4 extends Sample<int, Int32x4, SignalInt32x4, Scale<int>> {
  // ignore: non_constant_identifier_names
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
  // ignore: non_constant_identifier_names
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
      SampleFloat32x4.normalized(
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
  // ignore: non_constant_identifier_names
  static final RegExp REGEXP_IN_OUT_DELIMITER = RegExp(r'\s*=\s*');
// ignore: non_constant_identifier_names
  static final RegExp REGEXP_VALUE_DELIMITER = RegExp(r'\s*[,;]\s*');

  /// Input values.
  final T input;

  /// Output values.
  final T output;

  /// Scale of sample values.
  final S scale;

  Sample.normalized(this.input, this.output, this.scale);

  /// The signal level of the input. Used to sort samples.
  double get inputSignalLevel => input.computeSumSquaresMean();

  /// The signal level of the output. Used to sort samples.
  double get outputSignalLevel => output.computeSumSquaresMean();

  /// Normalize [signal] using [scale].
  T normalize(T signal) => signal.normalize(scale);

  /// Normalize [signal] using [scale].
  T normalizeWithScale(T signal, S scale) => signal.normalize(scale);

  /// Returns a [DataStatistics] of [input.values].
  DataStatistics inputStatistics() => input.statistics;

  /// Returns a [DataStatistics] of [output.values].
  DataStatistics outputStatistics() => output.statistics;

  DataStatistics inputProximityStatistics<P extends Sample<N, E, T, S>>(
      P other) {
    var values1 = input.values;
    var values2 = other.input.values;
    var diff = values1 - values2;
    return diff.statistics;
  }

  DataStatistics outputProximityStatistics<P extends Sample<N, E, T, S>>(
      P other) {
    var values1 = output.values;
    var values2 = other.output.values;
    var diff = values1 - values2;
    return diff.statistics;
  }

  DataStatistics proximityStatistics<P extends Sample<N, E, T, S>>(P other) {
    var input = inputProximityStatistics(other);
    var output = inputProximityStatistics(other);
    return DataStatistics(
      input.length,
      math.min(input.min, output.min),
      math.max(input.max, output.max),
      (input.center + output.center) / 2,
      sum: (input.sum + output.sum) / 2,
      squaresSum: (input.squaresSum + output.squaresSum) / 2,
      mean: (input.mean + output.mean) / 2,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Sample &&
          runtimeType == other.runtimeType &&
          scale == other.scale &&
          output == other.output &&
          input == other.input;

  @override
  int get hashCode => input.hashCode ^ output.hashCode ^ scale.hashCode;

  @override
  String toString() {
    return '$runtimeType{${input.valuesAsString} -> ${output.valuesAsString} ; $scale}';
  }
}

/// Samples Set.
class SamplesSet<P extends Sample<num, dynamic, dynamic, Scale<num>>> {
  /// The subject/title of this samples set.
  final String subject;

  /// The input tolerance. Used to compute [inputGroups].
  final double inputTolerance;

  /// The output tolerance. Used to compute [outputGroups].
  final double outputTolerance;

  final List<P> samples;

  SamplesSet(this.samples,
      {this.subject = '',
      this.inputTolerance = 0.01,
      this.outputTolerance = 0.01});

  /// Input length.
  int get inputLength => samples.first.input.length;

  /// Output length.
  int get outputLength => samples.first.output.length;

  /// Returns the first samle.
  P get first => samples.first;

  P operator [](int index) => samples[index];

  /// The number of input patterns/groups.
  int get inputGroups {
    var groups = samplesInputsGroups();
    return groups.length;
  }

  /// The number of output patterns/groups.
  int get outputGroups {
    var groups = samplesOutputsGroups();
    return groups.length;
  }

  /// The computed default target Global Error.
  double get defaultTargetGlobalError => outputTolerance / samples.length;

  double? _targetGlobalError;

  double get targetGlobalError =>
      _targetGlobalError ?? defaultTargetGlobalError;

  set targetGlobalError(double? value) {
    if (value != null) {
      if (value < 1.0E-13) value = 1.0E-13;
    }
    _targetGlobalError = value;
  }

  List<P> samplesCopy() => List<P>.from(samples);

  Map<P, double> inputsSignalLevels([List<P>? samples]) {
    samples ??= this.samples;
    return Map.fromEntries(samples.map((s) => MapEntry(s, s.inputSignalLevel)));
  }

  Map<P, double> outputsSignalLevels([List<P>? samples]) {
    samples ??= this.samples;
    return Map.fromEntries(
        samples.map((s) => MapEntry(s, s.outputSignalLevel)));
  }

  List<P> samplesSortedByInput() {
    var samples2 = samplesCopy();
    var samplesLevels = inputsSignalLevels(samples2);
    samples2.sort((a, b) => samplesLevels[a]!.compareTo(samplesLevels[b]!));
    return samples2;
  }

  List<P> samplesSortedByOutput() {
    var samples2 = samplesCopy();
    var samplesLevels = outputsSignalLevels(samples2);
    samples2.sort((a, b) => samplesLevels[a]!.compareTo(samplesLevels[b]!));
    return samples2;
  }

  int get length => samples.length;

  /// Computes the samples inputs groups.
  List<Set<P>> samplesInputsGroups({double? tolerance}) =>
      samplesSimilarityGroups((s1, s2) => s1.inputProximityStatistics(s2).mean,
          tolerance: tolerance ?? inputTolerance,
          samples: samplesSortedByInput());

  /// Computes the samples outputs groups.
  List<Set<P>> samplesOutputsGroups({double? tolerance}) =>
      samplesSimilarityGroups((s1, s2) => s1.outputProximityStatistics(s2).mean,
          tolerance: tolerance ?? outputTolerance,
          samples: samplesSortedByOutput());

  List<Set<P>> samplesSimilarityGroups(double Function(P s1, P s2) proximity,
      {double? tolerance, List<P>? samples}) {
    tolerance ??= inputTolerance;
    samples ??= this.samples;

    var length = this.length;

    var groups = <Set<P>>[];
    var samplesGroups = <P, int>{};

    for (var i = 0; i < length; ++i) {
      var s1 = samples[i];
      var groupIdx = samplesGroups[s1];
      var group = groupIdx != null ? groups[groupIdx] : null;

      var added = false;

      for (var j = i + 1; j < length; ++j) {
        var s2 = samples[j];
        var diff = proximity(s1, s2).abs();

        if (diff < tolerance) {
          if (group == null) {
            group = {s1, s2};
            groupIdx = groups.length;

            groups.add(group);
            samplesGroups[s1] = groupIdx;
            samplesGroups[s2] = groupIdx;
          } else if (!samplesGroups.containsKey(s2)) {
            group.add(s2);
            samplesGroups[s2] = groupIdx!;
          }

          added = true;
        }
      }

      if (!added && group == null) {
        group = {s1};
        groupIdx = groups.length;

        groups.add(group);
        samplesGroups[s1] = groupIdx;
      }
    }

    return groups;
  }

  Map<P, int> samplesGroupsIndexes(List<Set<P>> groups) {
    var map = <P, int>{};

    for (var i = groups.length - 1; i >= 0; --i) {
      var g = groups[i];
      for (var s in g) {
        map[s] = i;
      }
    }

    return map;
  }

  /// Compute all samples with conflicts.
  List<Map<int, List<P>>> computeConflicts(
      {double? inputTolerance, double? outputTolerance}) {
    return _computeConflictsImpl(
        inputTolerance: inputTolerance, outputTolerance: outputTolerance);
  }

  List<Map<int, List<P>>> _computeConflictsImpl(
      {double? inputTolerance,
      double? outputTolerance,
      List<Set<P>>? outputsGroups}) {
    var inputsGroups = samplesInputsGroups(tolerance: inputTolerance);
    outputsGroups ??= samplesOutputsGroups(tolerance: outputTolerance);

    var samplesOutputGroupsIdx = samplesGroupsIndexes(outputsGroups);

    var conflicts = <Map<int, List<P>>>[];

    for (var g in inputsGroups) {
      var outputGroups = g.groupBy((s) => samplesOutputGroupsIdx[s]!);
      if (outputGroups.length > 1) {
        conflicts.add(outputGroups);
      }
    }

    return conflicts;
  }

  /// Computs samples with conflicts that should be removed.
  List<P> computeConflictsToRemove(
      {double? inputTolerance, double? outputTolerance}) {
    var outputsGroups = samplesOutputsGroups(tolerance: outputTolerance);
    if (outputsGroups.length <= 1) return <P>[];

    var conflicts = _computeConflictsImpl(
        inputTolerance: inputTolerance,
        outputTolerance: outputTolerance,
        outputsGroups: outputsGroups);

    var outputsGroupsSizes = outputsGroups.map((g) => g.length).toList();

    var removed = <P>[];
    for (var outputGroups in conflicts) {
      var groupsIdx = outputGroups.keys.toList();
      groupsIdx.sort(
          (a, b) => outputsGroupsSizes[a].compareTo(outputsGroupsSizes[b]));

      var limit = groupsIdx.length - 1;
      for (var gIdx = 0; gIdx < limit; ++gIdx) {
        var g = outputGroups[gIdx]!;
        removed.addAll(g);
      }
    }

    return removed;
  }

  /// Removes samples with conflict (similar inputs with different output group).
  List<P> removeConflicts({double? inputTolerance, double? outputTolerance}) {
    var toRemove = computeConflictsToRemove(
        inputTolerance: inputTolerance, outputTolerance: outputTolerance);

    if (toRemove.isNotEmpty) {
      samples.removeWhere((e) => toRemove.contains(e));
    }

    return toRemove;
  }
}

/// Samples Generator.
class SamplesGenerator {
  /// The input scale.
  ScaleDouble inputScale;

  /// The output scale. Default to 0-1 ([ScaleDouble.ZERO_TO_ONE]).
  ScaleDouble outputScale;

  /// [Function] that generates the output value.
  double Function(double n) f;

  /// Number of samples to generate.
  int length;

  SamplesGenerator(this.inputScale, this.f, this.length,
      [ScaleDouble? outputScale])
      : outputScale = outputScale ?? ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> generateSamples({int stepSize = 1}) {
    if (stepSize < 1) stepSize = 1;

    if (stepSize == 1) {
      return List.generate(length + 1, generateSampleAtIndex);
    }

    var samples = <SampleFloat32x4>[];

    for (var i = 0; i <= length; i += stepSize) {
      var s = generateSampleAtIndex(i);
      samples.add(s);
    }

    return samples;
  }

  /// Generates the samples at [index].
  SampleFloat32x4 generateSampleAtIndex(int index) {
    var input = index / length;
    return generateSampleByInput(input);
  }

  /// Generates the sample for [input].
  SampleFloat32x4 generateSampleByInput(double input) {
    var x = inputScale.denormalize(input);
    var y = f(x);
    var output = outputScale.normalize(y);
    var sample = SampleFloat32x4.fromNormalized([input], [output], inputScale);
    return sample;
  }
}
