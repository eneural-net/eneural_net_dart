import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Differential tests: the native CPU/Metal backend must reproduce the
/// pure-Dart trainer's weights within float32 tolerance and converge on the
/// same problems.
///
/// When no native library is available (web, missing dylib, unsupported arch)
/// the whole group is skipped — the native trainers fall back to pure Dart and
/// the existing suites already cover that path.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({int hidden = 4, int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(hidden, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  // Probe native availability once.
  final probe = NativeRProp(
    build(),
    SamplesSet(xor(), subject: 'probe'),
    backend: NativeBackend.cpu,
  )..logEnabled = false;
  final nativeAvailable = probe.isNativeAccelerated;

  final metalProbe = NativeRProp(
    build(),
    SamplesSet(xor(), subject: 'probe'),
    backend: NativeBackend.metal,
  )..logEnabled = false;
  final metalAvailable =
      metalProbe.isNativeAccelerated &&
      metalProbe.activeBackend == NativeBackend.metal;

  double maxAbsDiff(List<double> a, List<double> b) {
    var m = 0.0;
    for (var i = 0; i < a.length; ++i) {
      final d = (a[i] - b[i]).abs();
      if (d > m) m = d;
    }
    return m;
  }

  group(
    'Native CPU acceleration (differential vs pure Dart)',
    () {
      test('reports native backend active', () {
        expect(probe.activeBackend, equals(NativeBackend.cpu));
      });

      test('native forward pass (enn_activate) matches Dart activate', () {
        final t = NativeRProp(
          build(seed: 33),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.cpu,
        )..logEnabled = false;

        // Untrained random weights: compare inference on every XOR input.
        for (final s in xor()) {
          t.ann.activate(s.input);
          final dartOut = t.ann.outputAsDouble;
          final nativeOut = t.activateNative(s.input);
          expect(nativeOut, isNotNull);
          expect(
            maxAbsDiff(dartOut, nativeOut!),
            lessThan(1e-5),
            reason: 'forward output for ${s.input.values}',
          );
        }
      });

      test('single Backpropagation epoch matches Dart weights', () {
        final dartTrainer = Backpropagation(
          build(seed: 7),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;
        final nativeTrainer = NativeBackpropagation(
          build(seed: 7),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.cpu,
        )..logEnabled = false;

        // Same seed -> identical initial weights.
        expect(
          maxAbsDiff(
            dartTrainer.ann.allWeights.cast<double>(),
            nativeTrainer.ann.allWeights.cast<double>(),
          ),
          equals(0.0),
        );

        dartTrainer.train(1, 0.0);
        nativeTrainer.train(1, 0.0);

        final diff = maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          nativeTrainer.ann.allWeights.cast<double>(),
        );
        expect(diff, lessThan(1e-4), reason: 'max |Δweight| after 1 BP epoch');
      });

      test('several Backpropagation epochs stay close to Dart weights', () {
        final dartTrainer = Backpropagation(
          build(seed: 21),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;
        final nativeTrainer = NativeBackpropagation(
          build(seed: 21),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.cpu,
        )..logEnabled = false;

        dartTrainer.train(20, 0.0);
        nativeTrainer.train(20, 0.0);

        final diff = maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          nativeTrainer.ann.allWeights.cast<double>(),
        );
        expect(
          diff,
          lessThan(1e-3),
          reason: 'max |Δweight| after 20 BP epochs',
        );
      });

      test('native iRProp+ converges on XOR', () {
        final t = NativeRProp(
          build(seed: 101),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.cpu,
        )..logEnabled = false;

        final ok = t.trainUntilGlobalError(
          targetGlobalError: 1e-6,
          maxEpochs: 2000,
        );

        expect(ok, isTrue);
        expect(t.globalError, lessThan(1e-6));

        // Verify the learned function.
        final ann = t.ann;
        for (final s in xor()) {
          ann.activate(s.input);
          final out = ann.outputAsDouble.first;
          final expected = s.output.valuesAsDouble.first;
          expect((out - expected).abs(), lessThan(0.05));
        }
      });

      test('native and Dart iRProp+ both converge below target', () {
        final dartTrainer = NativeRProp(
          build(seed: 55),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.none, // pure Dart
        )..logEnabled = false;
        final nativeTrainer = NativeRProp(
          build(seed: 55),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.cpu,
        )..logEnabled = false;

        final okDart = dartTrainer.trainUntilGlobalError(
          targetGlobalError: 1e-6,
          maxEpochs: 3000,
        );
        final okNative = nativeTrainer.trainUntilGlobalError(
          targetGlobalError: 1e-6,
          maxEpochs: 3000,
        );

        expect(okDart, isTrue);
        expect(okNative, isTrue);
        expect(dartTrainer.globalError, lessThan(1e-6));
        expect(nativeTrainer.globalError, lessThan(1e-6));
      });
    },
    skip: nativeAvailable ? false : 'native library not available',
  );

  group(
    'Native Metal acceleration (differential vs pure Dart)',
    () {
      test('reports metal backend active', () {
        expect(metalProbe.activeBackend, equals(NativeBackend.metal));
      });

      test('native forward pass matches Dart activate', () {
        final t = NativeRProp(
          build(seed: 33),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.metal,
        )..logEnabled = false;
        for (final s in xor()) {
          t.ann.activate(s.input);
          final dartOut = t.ann.outputAsDouble;
          final nativeOut = t.activateNative(s.input);
          expect(nativeOut, isNotNull);
          expect(
            maxAbsDiff(dartOut, nativeOut!),
            lessThan(1e-4),
            reason: 'forward output for ${s.input.values}',
          );
        }
      });

      test('single Backpropagation epoch matches Dart weights', () {
        final dartTrainer = Backpropagation(
          build(seed: 7),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;
        final nativeTrainer = NativeBackpropagation(
          build(seed: 7),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.metal,
        )..logEnabled = false;

        dartTrainer.train(1, 0.0);
        nativeTrainer.train(1, 0.0);

        final diff = maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          nativeTrainer.ann.allWeights.cast<double>(),
        );
        expect(diff, lessThan(1e-4), reason: 'max |Δweight| after 1 BP epoch');
      });

      test('native iRProp+ converges on XOR', () {
        final t = NativeRProp(
          build(seed: 101),
          SamplesSet(xor(), subject: 'xor'),
          backend: NativeBackend.metal,
        )..logEnabled = false;

        final ok = t.trainUntilGlobalError(
          targetGlobalError: 1e-6,
          maxEpochs: 3000,
        );

        expect(ok, isTrue);
        expect(t.globalError, lessThan(1e-6));
        for (final s in xor()) {
          t.ann.activate(s.input);
          final out = t.ann.outputAsDouble.first;
          final expected = s.output.valuesAsDouble.first;
          expect((out - expected).abs(), lessThan(0.05));
        }
      });
    },
    skip: metalAvailable ? false : 'metal backend not available',
  );

  group('Native trainer fallback (backend none == pure Dart)', () {
    test('backend none is not native-accelerated', () {
      final t = NativeRProp(
        build(),
        SamplesSet(xor(), subject: 'xor'),
        backend: NativeBackend.none,
      )..logEnabled = false;
      expect(t.isNativeAccelerated, isFalse);
      expect(t.activeBackend, equals(NativeBackend.none));
    });

    test('backend none reproduces pure-Dart RProp exactly', () {
      final ref = RProp(build(seed: 9), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;
      final fallback = NativeRProp(
        build(seed: 9),
        SamplesSet(xor(), subject: 'xor'),
        backend: NativeBackend.none,
      )..logEnabled = false;

      ref.train(30, 0.0);
      fallback.train(30, 0.0);

      final maxDiff = () {
        final a = ref.ann.allWeights.cast<double>();
        final b = fallback.ann.allWeights.cast<double>();
        var m = 0.0;
        for (var i = 0; i < a.length; ++i) {
          final d = (a[i] - b[i]).abs();
          if (d > m) m = d;
        }
        return m;
      }();

      expect(maxDiff, equals(0.0));
    });
  });
}
