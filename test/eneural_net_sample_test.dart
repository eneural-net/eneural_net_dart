import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;
  var scaleInt = ScaleInt.ZERO_TO_ONE;

  List<SampleFloat32x4> xorSamples() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scaleDouble,
    true,
  );

  group('SampleFloat32x4: construction', () {
    test('fromNormalized keeps the values as they are', () {
      var s = SampleFloat32x4.fromNormalized([0.25, 0.75], [1.0], scaleDouble);

      expect(s.input.values, equals([0.25, 0.75]));
      expect(s.output.values, equals([1.0]));
      expect(s.scale, equals(scaleDouble));
    });

    test('from normalizes with the scale', () {
      var scale = ScaleDouble(0, 100);
      var s = SampleFloat32x4.from([0, 50, 100], [100], scale);

      expect(s.input.values, equals([0.0, 0.5, 1.0]));
      expect(s.output.values, equals([1.0]));
    });

    test('the default constructor normalizes signals', () {
      var scale = ScaleDouble(0, 100);
      var s = SampleFloat32x4(
        SignalFloat32x4.from([0, 50, 100]),
        SignalFloat32x4.from([100]),
        scale,
      );

      expect(s.input.values, equals([0.0, 0.5, 1.0]));
      expect(s.output.values, equals([1.0]));
    });

    test('normalized wraps the given signals', () {
      var input = SignalFloat32x4.from([1, 2]);
      var output = SignalFloat32x4.from([3]);
      var s = SampleFloat32x4.normalized(input, output, scaleDouble);

      expect(s.input, same(input));
      expect(s.output, same(output));
    });

    test('fromString parses both delimiters', () {
      var s1 = SampleFloat32x4.fromString('0,1=1', scaleDouble, true);
      expect(s1.input.values, equals([0.0, 1.0]));
      expect(s1.output.values, equals([1.0]));

      var s2 = SampleFloat32x4.fromString('0;1 = 1;0', scaleDouble, true);
      expect(s2.input.values, equals([0.0, 1.0]));
      expect(s2.output.values, equals([1.0, 0.0]));

      var s3 = SampleFloat32x4.fromString(' 0 , 1 = 1 ', scaleDouble, true);
      expect(s3.input.values, equals([0.0, 1.0]));
    });

    test('fromString can denormalize', () {
      var scale = ScaleDouble(0, 100);
      var s = SampleFloat32x4.fromString('0,100=100', scale, false);

      expect(s.input.values, equals([0.0, 1.0]));
      expect(s.output.values, equals([1.0]));
    });

    test('toList from pairs', () {
      var samples = SampleFloat32x4.toList([
        [
          [0, 0],
          [0],
        ],
        [
          [1, 1],
          [1],
        ],
      ], scaleDouble);

      expect(samples.length, equals(2));
      expect(samples[0].input.values, equals([0.0, 0.0]));
      expect(samples[1].output.values, equals([1.0]));
    });

    test('toListFromString', () {
      var samples = xorSamples();

      expect(samples.length, equals(4));
      expect(samples[0].input.values, equals([0.0, 0.0]));
      expect(samples[3].input.values, equals([1.0, 1.0]));
      expect(samples[3].output.values, equals([0.0]));
    });

    test('DUMMY is usable', () {
      expect(SampleFloat32x4.DUMMY.input.length, equals(1));
      expect(SampleInt32x4.DUMMY.input.length, equals(1));
    });
  });

  group('SampleInt32x4: construction', () {
    test('fromNormalized/from/fromString', () {
      var s = SampleInt32x4.fromNormalized([0, 1], [1], scaleInt);
      expect(s.input.values, equals([0, 1]));

      var s2 = SampleInt32x4.from([0, 1], [1], scaleInt);
      expect(s2.input.values, equals([0, 1]));

      var s3 = SampleInt32x4.fromString('0,1=1', scaleInt, true);
      expect(s3.input.values, equals([0, 1]));
      expect(s3.output.values, equals([1]));

      var s4 = SampleInt32x4.fromString('0,1=1', scaleInt, false);
      expect(s4.input.values, equals([0, 1]));
    });

    test('the default constructor normalizes signals', () {
      var s = SampleInt32x4(
        SignalInt32x4.from([0, 1]),
        SignalInt32x4.from([1]),
        scaleInt,
      );
      expect(s.input.values, equals([0, 1]));
    });

    test('toList/toListFromString', () {
      var samples = SampleInt32x4.toList([
        [
          [0, 0],
          [0],
        ],
      ], scaleInt);
      expect(samples.length, equals(1));

      var samples2 = SampleInt32x4.toListFromString(
        ['0,0=0', '1,1=0'],
        scaleInt,
        true,
      );
      expect(samples2.length, equals(2));
    });
  });

  group('Sample: behavior', () {
    test('signal levels', () {
      var s = SampleFloat32x4.fromNormalized([1.0, 1.0], [1.0], scaleDouble);

      expect(s.inputSignalLevel, equals(1.0));
      expect(s.outputSignalLevel, equals(1.0));

      var zero = SampleFloat32x4.fromNormalized([0.0, 0.0], [0.0], scaleDouble);
      expect(zero.inputSignalLevel, equals(0.0));
    });

    test('normalize/normalizeWithScale', () {
      var scale = ScaleDouble(0, 100);
      var s = SampleFloat32x4.fromNormalized([0.0], [0.0], scale);

      expect(s.normalize(SignalFloat32x4.from([50])).values, equals([0.5]));
      expect(
        s
            .normalizeWithScale(SignalFloat32x4.from([50]), ScaleDouble(0, 200))
            .values,
        equals([0.25]),
      );
    });

    test('input/output statistics', () {
      var s = SampleFloat32x4.fromNormalized(
        [0.0, 0.5, 1.0],
        [1.0],
        scaleDouble,
      );

      expect(s.inputStatistics().mean, closeTo(0.5, 1e-6));
      expect(s.outputStatistics().mean, equals(1.0));
    });

    test('input/output proximity statistics', () {
      var a = SampleFloat32x4.fromNormalized([0.0, 0.0], [0.0], scaleDouble);
      var b = SampleFloat32x4.fromNormalized([0.0, 0.0], [1.0], scaleDouble);

      expect(a.inputProximityStatistics(b).mean, equals(0.0));
      expect(a.outputProximityStatistics(b).mean, equals(-1.0));
    });

    test('proximityStatistics combines input AND output', () {
      var a = SampleFloat32x4.fromNormalized([0.0, 0.0], [0.0], scaleDouble);
      var b = SampleFloat32x4.fromNormalized([0.0, 0.0], [1.0], scaleDouble);

      var proximity = a.proximityStatistics(b);

      // Same inputs (0) but different outputs (-1): the combination must not
      // be the input proximity alone.
      expect(proximity.mean, closeTo(-0.5, 1e-9));
      expect(proximity.min, equals(-1.0));
      expect(proximity.max, equals(0.0));
    });

    test('proximityStatistics of identical samples is zero', () {
      var a = SampleFloat32x4.fromNormalized([0.5, 0.5], [0.5], scaleDouble);
      var b = SampleFloat32x4.fromNormalized([0.5, 0.5], [0.5], scaleDouble);

      expect(a.proximityStatistics(b).mean, equals(0.0));
    });

    test('equality', () {
      var a = SampleFloat32x4.fromNormalized([0.0, 1.0], [1.0], scaleDouble);
      var b = SampleFloat32x4.fromNormalized([0.0, 1.0], [1.0], scaleDouble);
      var c = SampleFloat32x4.fromNormalized([1.0, 1.0], [1.0], scaleDouble);

      expect(a, equals(b));
      expect(a.hashCode, equals(b.hashCode));
      expect(a, isNot(equals(c)));
      expect(a, equals(a));
      expect(a, isNot(equals('not a sample')));
    });

    test('different scales are not equal', () {
      var a = SampleFloat32x4.fromNormalized([0.0], [0.0], ScaleDouble(0, 1));
      var b = SampleFloat32x4.fromNormalized([0.0], [0.0], ScaleDouble(0, 2));

      expect(a, isNot(equals(b)));
    });

    test('toString', () {
      var s = SampleFloat32x4.fromNormalized([0.0, 1.0], [1.0], scaleDouble);

      expect(s.toString(), contains('SampleFloat32x4'));
      expect(s.toString(), contains('->'));
      expect(s.toString(), contains('ScaleDouble'));
    });
  });

  group('SamplesSet', () {
    test('basic properties', () {
      var set = SamplesSet(xorSamples(), subject: 'xor');

      expect(set.subject, equals('xor'));
      expect(set.length, equals(4));
      expect(set.inputLength, equals(2));
      expect(set.outputLength, equals(1));
      expect(set.inputTolerance, equals(0.01));
      expect(set.outputTolerance, equals(0.01));
      expect(set.first, equals(set[0]));
      expect(set[3].input.values, equals([1.0, 1.0]));
    });

    test('samplesCopy is a new list with the same samples', () {
      var samples = xorSamples();
      var set = SamplesSet(samples);
      var copy = set.samplesCopy();

      expect(copy, equals(samples));
      expect(identical(copy, samples), isFalse);
    });

    test('targetGlobalError', () {
      var set = SamplesSet(xorSamples());

      expect(set.defaultTargetGlobalError, equals(0.01 / 4));
      expect(set.targetGlobalError, equals(set.defaultTargetGlobalError));

      set.targetGlobalError = 0.5;
      expect(set.targetGlobalError, equals(0.5));

      // Clamped to a minimum:
      set.targetGlobalError = 0.0;
      expect(set.targetGlobalError, equals(1.0E-13));

      set.targetGlobalError = null;
      expect(set.targetGlobalError, equals(set.defaultTargetGlobalError));
    });

    test('signal levels maps', () {
      var set = SamplesSet(xorSamples());

      expect(set.inputsSignalLevels().length, equals(4));
      expect(set.outputsSignalLevels().length, equals(4));
      expect(
        set.inputsSignalLevels(set.samples.sublist(0, 2)).length,
        equals(2),
      );
      expect(
        set.outputsSignalLevels(set.samples.sublist(0, 2)).length,
        equals(2),
      );
    });

    test('samplesSortedByInput/Output', () {
      var set = SamplesSet(xorSamples());

      var byInput = set.samplesSortedByInput();
      expect(byInput.length, equals(4));
      expect(
        byInput.first.inputSignalLevel <= byInput.last.inputSignalLevel,
        isTrue,
      );

      var byOutput = set.samplesSortedByOutput();
      expect(
        byOutput.first.outputSignalLevel <= byOutput.last.outputSignalLevel,
        isTrue,
      );

      // The original set is not modified:
      expect(set.samples.first.input.values, equals([0.0, 0.0]));
    });

    test('input and output groups of the XOR set', () {
      var set = SamplesSet(xorSamples());

      // Two distinct outputs (0 and 1):
      expect(set.outputGroups, equals(2));
      // Four distinct inputs:
      expect(set.inputGroups >= 1, isTrue);
    });

    test('samplesSimilarityGroups honors the given samples list', () {
      var set = SamplesSet(xorSamples());

      var groups = set.samplesSimilarityGroups(
        (s1, s2) => s1.inputProximityStatistics(s2).mean,
        samples: set.samples.sublist(0, 2),
      );

      expect(groups, isNotEmpty);
      expect(
        groups.expand((g) => g).toSet().length <= 2,
        isTrue,
        reason: 'only the given samples can be grouped',
      );
    });

    test('samplesSimilarityGroups with a huge tolerance groups everything', () {
      var set = SamplesSet(xorSamples());

      var groups = set.samplesSimilarityGroups(
        (s1, s2) => s1.inputProximityStatistics(s2).mean,
        tolerance: 1000,
      );

      expect(groups.length, equals(1));
      expect(groups.first.length, equals(4));
    });

    test('samplesGroupsIndexes maps each sample to its group', () {
      var set = SamplesSet(xorSamples());
      var groups = set.samplesOutputsGroups();
      var indexes = set.samplesGroupsIndexes(groups);

      for (var i = 0; i < groups.length; ++i) {
        for (var s in groups[i]) {
          expect(indexes[s], equals(i));
        }
      }
    });

    test('computeConflicts finds none in a consistent set', () {
      var set = SamplesSet(xorSamples());
      expect(set.computeConflicts(), isEmpty);
      expect(set.computeConflictsToRemove(), isEmpty);
    });

    test('computeConflicts finds contradictory samples', () {
      // Same input mapped to two different outputs:
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '0,0=1', '1,1=1', '1,1=1'],
        scaleDouble,
        true,
      );

      var set = SamplesSet(samples);
      var conflicts = set.computeConflicts();

      expect(conflicts, isNotEmpty);
    });

    test('removeConflicts removes samples from the set', () {
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '0,0=1', '1,1=1', '1,1=1'],
        scaleDouble,
        true,
      );

      var set = SamplesSet(samples);
      var initialLength = set.length;
      var removed = set.removeConflicts();

      expect(set.length, equals(initialLength - removed.length));
    });

    test('removeConflicts of a single-output set removes nothing', () {
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '1,1=0'],
        scaleDouble,
        true,
      );

      var set = SamplesSet(samples);
      expect(set.removeConflicts(), isEmpty);
      expect(set.length, equals(2));
    });
  });

  group('SamplesGenerator', () {
    test('generates length+1 samples', () {
      var gen = SamplesGenerator(ScaleDouble(0, 10), (x) => x, 10);
      var samples = gen.generateSamples();

      expect(samples.length, equals(11));
      expect(gen.length, equals(10));
      expect(gen.inputScale, equals(ScaleDouble(0, 10)));
      expect(gen.outputScale, equals(ScaleDouble.ZERO_TO_ONE));
    });

    test('generateSampleAtIndex maps the index to a normalized input', () {
      var gen = SamplesGenerator(ScaleDouble(0, 10), (x) => x / 10, 10);

      expect(gen.generateSampleAtIndex(0).input.values, equals([0.0]));
      expect(gen.generateSampleAtIndex(10).input.values, equals([1.0]));
      expect(gen.generateSampleAtIndex(5).input.values, equals([0.5]));
    });

    test('generateSampleByInput applies the function to the scaled input', () {
      // f(x) = x / 100 over the input scale 0..100 -> output in 0..1.
      var gen = SamplesGenerator(ScaleDouble(0, 100), (x) => x / 100, 10);

      var s = gen.generateSampleByInput(0.5);
      expect(s.input.values, equals([0.5]));
      expect(s.output.values.first, closeTo(0.5, 1e-6));
    });

    test('stepSize skips samples', () {
      var gen = SamplesGenerator(ScaleDouble(0, 10), (x) => x / 10, 10);

      expect(gen.generateSamples(stepSize: 2).length, equals(6));
      expect(gen.generateSamples(stepSize: 5).length, equals(3));
      expect(
        gen.generateSamples(stepSize: 0).length,
        equals(11),
        reason: 'a step below 1 is treated as 1',
      );
    });

    test('a custom output scale is applied', () {
      var gen = SamplesGenerator(
        ScaleDouble(0, 10),
        (x) => x,
        10,
        ScaleDouble(0, 10),
      );

      expect(gen.outputScale, equals(ScaleDouble(0, 10)));

      var s = gen.generateSampleByInput(1.0);
      expect(s.output.values.first, closeTo(1.0, 1e-6));
    });

    test('generates a trainable sinusoidal-like series', () {
      var gen = SamplesGenerator(ScaleDouble(0, 1), (x) => x * x, 20);
      var samples = gen.generateSamples();

      expect(samples.length, equals(21));
      expect(samples.every((s) => s.input.length == 1), isTrue);
      expect(samples.every((s) => s.output.length == 1), isTrue);
    });
  });
}
