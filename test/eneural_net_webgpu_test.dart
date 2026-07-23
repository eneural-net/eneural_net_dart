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
}
