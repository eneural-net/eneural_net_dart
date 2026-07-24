import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Tests for the WebGPU async trainers.
///
/// On the Dart VM (and any browser without WebGPU) the trainers fall back to the
/// synchronous pure-Dart path, so these tests run everywhere. On a browser with
/// WebGPU the same assertions exercise the GPU path.
///
/// The GPU path itself (WGSL shaders, buffer transfers, differential checks
/// against the pure-Dart trainer) is covered by the browser-only
/// `eneural_net_webgpu_integration_test.dart`.
void main() {
  /// Whether this suite was compiled for the web (where WebGPU may exist).
  const isWeb = bool.fromEnvironment('dart.library.js_interop');

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

  group('WebGpu trainers (GPU or pure-Dart fallback)', () {
    test('isWebGpuAccelerated resolves to a bool', () async {
      final t = WebGpuRProp(build(), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;
      final accelerated = await t.isWebGpuAccelerated;
      expect(accelerated, isA<bool>());
    });

    test('WebGpuRProp.trainUntilGlobalErrorAsync converges on XOR', () async {
      final t = WebGpuRProp(build(seed: 101), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;

      final ok = await t.trainUntilGlobalErrorAsync(
        targetGlobalError: 1e-4,
        maxEpochs: 5000,
      );

      expect(ok, isTrue);
      expect(t.globalError, lessThan(1e-4));

      final ann = t.ann;
      for (final s in xor()) {
        ann.activate(s.input);
        final out = ann.outputAsDouble.first;
        final expected = s.output.valuesAsDouble.first;
        expect((out - expected).abs(), lessThan(0.1));
      }
    });

    test('WebGpuBackpropagation.trainAsync reduces the error', () async {
      final t = WebGpuBackpropagation(
        build(seed: 7),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      final before = t.ann.computeSamplesGlobalError(xor());
      final after = await t.trainAsync(200, 0.0);

      expect(after, lessThan(before));
    });
  });

  group('WebGpu pure-Dart fallback', () {
    test('the Dart VM is never WebGPU accelerated', () async {
      if (isWeb) {
        markTestSkipped('web platform: WebGPU may be available');
        return;
      }

      final t = WebGpuRProp(build(), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;
      expect(await t.isWebGpuAccelerated, isFalse);
    });

    test('the fallback reproduces the pure-Dart RProp exactly', () async {
      if (isWeb) {
        markTestSkipped('web platform: WebGPU may be available');
        return;
      }

      final ref = RProp(build(seed: 9), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;
      final fallback = WebGpuRProp(
        build(seed: 9),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      ref.train(30, 0.0);
      await fallback.trainAsync(30, 0.0);

      final a = ref.ann.allWeights.cast<double>();
      final b = fallback.ann.allWeights.cast<double>();
      var maxDiff = 0.0;
      for (var i = 0; i < a.length; ++i) {
        final d = (a[i] - b[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }

      expect(maxDiff, equals(0.0));
    });

    test('activateWebGpu returns null without a WebGPU device', () async {
      if (isWeb) {
        markTestSkipped('web platform: WebGPU may be available');
        return;
      }

      final t = WebGpuRProp(build(), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;
      expect(await t.activateWebGpu(xor().first.input), isNull);
    });
  });
}
