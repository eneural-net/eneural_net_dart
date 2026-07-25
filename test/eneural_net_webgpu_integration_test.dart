@TestOn('browser')
@Tags(['webgpu'])
library;

import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// When compiled with `-DWEBGPU_REQUIRED=true` a browser without a usable
/// WebGPU device is a FAILURE instead of a skip.
///
/// CI uses it so that a job cannot pass vacuously through the pure-Dart
/// fallback (which the VM suite already covers):
///
///   dart test test/eneural_net_webgpu_integration_test.dart \
///     --platform chrome_webgpu --dart2js-args=-DWEBGPU_REQUIRED=true
const requireWebGpu = bool.fromEnvironment('WEBGPU_REQUIRED');

/// WebGPU integration tests: these exercise the real GPU path (WGSL compute
/// shaders through `dart:js_interop`), not the pure-Dart fallback.
///
/// They run in a browser only (`@TestOn('browser')`) and self-skip when the
/// browser exposes no WebGPU device, unless [requireWebGpu] is set.
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

  /// Random samples for the larger-network tests (inputs/outputs already in the
  /// `0..1` scale).
  List<SampleFloat32x4> randomSamples(
    int count,
    int inputSize,
    int outputSize, {
    int seed = 7,
  }) {
    final r = Random(seed);
    return List.generate(
      count,
      (_) => SampleFloat32x4.fromNormalized(
        List.generate(inputSize, (_) => r.nextDouble()),
        List.generate(outputSize, (_) => r.nextDouble()),
        scale,
      ),
    );
  }

  double maxAbsDiff(List<double> a, List<double> b) {
    var m = 0.0;
    for (var i = 0; i < a.length; ++i) {
      final d = (a[i] - b[i]).abs();
      if (d > m) m = d;
    }
    return m;
  }

  // Probed once: whether this browser actually resolved a WebGPU device.
  var gpuAvailable = false;

  setUpAll(() async {
    final probe = WebGpuRProp(build(), SamplesSet(xor(), subject: 'probe'))
      ..logEnabled = false;
    gpuAvailable = await probe.isWebGpuAccelerated;
  });

  /// Returns `true` when the GPU path is active. Otherwise fails (strict CI) or
  /// marks the running test as skipped.
  bool onGpu() {
    if (gpuAvailable) return true;
    if (requireWebGpu) {
      fail(
        'No WebGPU device: compiled with -DWEBGPU_REQUIRED=true, so the '
        'pure-Dart fallback is not accepted here.',
      );
    }
    markTestSkipped('WebGPU not available in this browser');
    return false;
  }

  group('WebGPU device', () {
    test('resolves (or reports the absence of) a WebGPU device', () async {
      final t = WebGpuRProp(build(), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;

      final accelerated = await t.isWebGpuAccelerated;
      expect(accelerated, isA<bool>());

      if (requireWebGpu) {
        expect(
          accelerated,
          isTrue,
          reason: 'WEBGPU_REQUIRED=true but no WebGPU device was resolved',
        );
      }
    });

    test('the same trainer resolves the device only once', () async {
      if (!onGpu()) return;

      final t = WebGpuRProp(build(), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;

      expect(await t.isWebGpuAccelerated, isTrue);
      expect(await t.isWebGpuAccelerated, isTrue);
    });
  });

  group('WebGPU inference (GPU forward pass vs pure Dart)', () {
    // Topologies covering: bias / no bias, one / two hidden layers, each
    // supported activation function, and layer sizes above the shader
    // workgroup size (8x8) so multiple workgroups are dispatched.
    final topologies = <String, ANNF Function()>{
      '2-4-1 sigmoid, bias': () => build(seed: 33),
      '2-6-1 sigmoid, no bias': () => ANN(
        scale,
        LayerFloat32x4(2, false),
        [HiddenLayerConfig(6, false)],
        LayerFloat32x4(1, false),
        random: Random(11),
      ),
      '3-10-7-2 two hidden layers': () => ANN(
        scale,
        LayerFloat32x4(3, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(10, true), HiddenLayerConfig(7, true)],
        LayerFloat32x4(2, false),
        random: Random(13),
      ),
      '4-9-3 linear output': () => ANN(
        scale,
        LayerFloat32x4(4, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(9, true)],
        LayerFloat32x4(3, false, ActivationFunctionLinear()),
        random: Random(17),
      ),
      '3-12-2 SigmoidFast': () => ANN(
        scale,
        LayerFloat32x4(3, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(12, true, ActivationFunctionSigmoidFast())],
        LayerFloat32x4(2, false, ActivationFunctionSigmoidFast()),
        random: Random(19),
      ),
      '3-12-2 SigmoidBoundedFast': () => ANN(
        scale,
        LayerFloat32x4(3, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(12, true, ActivationFunctionSigmoidBoundedFast())],
        LayerFloat32x4(2, false, ActivationFunctionSigmoidBoundedFast()),
        random: Random(23),
      ),
    };

    for (final entry in topologies.entries) {
      test('GPU activate matches Dart activate — ${entry.key}', () async {
        if (!onGpu()) return;

        final ann = entry.value();
        final inputSize =
            ann.inputLayer.length - (ann.inputLayer.withBiasNeuron ? 1 : 0);

        // A samples set is required to build the trainer, but inference does
        // not depend on it.
        final samples = randomSamples(
          4,
          inputSize,
          ann.outputLayer.length,
          seed: 3,
        );
        final t = WebGpuRProp(ann, SamplesSet(samples, subject: 'inference'))
          ..logEnabled = false;

        expect(await t.isWebGpuAccelerated, isTrue);

        final r = Random(5);
        for (var i = 0; i < 5; ++i) {
          final input = SignalFloat32x4.from(
            List.generate(inputSize, (_) => r.nextDouble()),
          );

          ann.activate(input);
          final dartOut = ann.outputAsDouble;

          final gpuOut = await t.activateWebGpu(input);
          expect(gpuOut, isNotNull);
          expect(gpuOut!.length, equals(dartOut.length));

          expect(
            maxAbsDiff(dartOut, gpuOut),
            lessThan(1e-4),
            reason: 'GPU vs Dart forward pass for input ${input.values}',
          );
        }
      });
    }
  });

  group('WebGPU weight transfer', () {
    test('weights survive a GPU upload/download round-trip', () async {
      if (!onGpu()) return;

      final t = WebGpuBackpropagation(
        build(seed: 41),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      expect(await t.isWebGpuAccelerated, isTrue);

      final before = t.ann.allWeights.cast<double>().toList();

      // Zero epochs: uploads the weights to the GPU buffers and reads them
      // back, exercising the per-layer buffer segmentation/ordering.
      await t.trainAsync(0, 0.0);

      final after = t.ann.allWeights.cast<double>().toList();

      expect(after.length, equals(before.length));
      expect(
        maxAbsDiff(before, after),
        equals(0.0),
        reason: 'float32 weights must round-trip exactly',
      );
    });

    test('trained GPU weights are written back into the ANN', () async {
      if (!onGpu()) return;

      final t = WebGpuRProp(build(seed: 43), SamplesSet(xor(), subject: 'xor'))
        ..logEnabled = false;

      expect(await t.isWebGpuAccelerated, isTrue);

      final before = t.ann.allWeights.cast<double>().toList();
      await t.trainAsync(20, 0.0);
      final after = t.ann.allWeights.cast<double>().toList();

      expect(
        maxAbsDiff(before, after),
        greaterThan(0.0),
        reason: 'training on the GPU must change the ANN weights',
      );

      // The ANN must agree with the GPU on the weights it now holds.
      for (final s in xor()) {
        t.ann.activate(s.input);
        final dartOut = t.ann.outputAsDouble;
        final gpuOut = await t.activateWebGpu(s.input);
        expect(maxAbsDiff(dartOut, gpuOut!), lessThan(1e-4));
      }
    });
  });

  group('WebGPU training (differential vs pure Dart)', () {
    test('a single Backpropagation epoch matches the Dart weights', () async {
      if (!onGpu()) return;

      final dartTrainer = Backpropagation(
        build(seed: 7),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;
      final gpuTrainer = WebGpuBackpropagation(
        build(seed: 7),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      // Same seed -> identical initial weights.
      expect(
        maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          gpuTrainer.ann.allWeights.cast<double>(),
        ),
        equals(0.0),
      );

      dartTrainer.train(1, 0.0);
      await gpuTrainer.trainAsync(1, 0.0);

      expect(
        maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          gpuTrainer.ann.allWeights.cast<double>(),
        ),
        lessThan(1e-4),
        reason: 'max |Δweight| after 1 Backpropagation epoch',
      );
    });

    test('20 Backpropagation epochs stay close to the Dart weights', () async {
      if (!onGpu()) return;

      final dartTrainer = Backpropagation(
        build(seed: 21),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;
      final gpuTrainer = WebGpuBackpropagation(
        build(seed: 21),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      dartTrainer.train(20, 0.0);
      await gpuTrainer.trainAsync(20, 0.0);

      expect(
        maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          gpuTrainer.ann.allWeights.cast<double>(),
        ),
        lessThan(1e-3),
        reason: 'max |Δweight| after 20 Backpropagation epochs',
      );
    });

    test('a single iRProp+ epoch matches the Dart weights', () async {
      if (!onGpu()) return;

      final dartTrainer = RProp(
        build(seed: 31),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;
      final gpuTrainer = WebGpuRProp(
        build(seed: 31),
        SamplesSet(xor(), subject: 'xor'),
      )..logEnabled = false;

      dartTrainer.train(1, 0.0);
      await gpuTrainer.trainAsync(1, 0.0);

      expect(
        maxAbsDiff(
          dartTrainer.ann.allWeights.cast<double>(),
          gpuTrainer.ann.allWeights.cast<double>(),
        ),
        lessThan(1e-4),
        reason: 'max |Δweight| after 1 iRProp+ epoch',
      );
    });

    test(
      'iRProp+ converges on XOR on the GPU',
      () async {
        if (!onGpu()) return;

        final t = WebGpuRProp(
          build(seed: 101),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;

        final ok = await t.trainUntilGlobalErrorAsync(
          targetGlobalError: 1e-5,
          maxEpochs: 5000,
        );

        expect(ok, isTrue);
        expect(t.globalError, lessThan(1e-5));

        for (final s in xor()) {
          t.ann.activate(s.input);
          final out = t.ann.outputAsDouble.first;
          final expectedOut = s.output.valuesAsDouble.first;
          expect((out - expectedOut).abs(), lessThan(0.05));
        }
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'GPU and Dart iRProp+ both converge below the target',
      () async {
        if (!onGpu()) return;

        final dartTrainer = RProp(
          build(seed: 55),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;
        final gpuTrainer = WebGpuRProp(
          build(seed: 55),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;

        final okDart = dartTrainer.trainUntilGlobalError(
          targetGlobalError: 1e-5,
          maxEpochs: 3000,
        );
        final okGpu = await gpuTrainer.trainUntilGlobalErrorAsync(
          targetGlobalError: 1e-5,
          maxEpochs: 3000,
        );

        expect(okDart, isTrue);
        expect(okGpu, isTrue);
        expect(dartTrainer.globalError, lessThan(1e-5));
        expect(gpuTrainer.globalError, lessThan(1e-5));
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );

    test(
      'Backpropagation on the GPU reduces the error',
      () async {
        if (!onGpu()) return;

        final t = WebGpuBackpropagation(
          build(seed: 7),
          SamplesSet(xor(), subject: 'xor'),
        )..logEnabled = false;

        final before = t.ann.computeSamplesGlobalError(xor());
        final after = await t.trainAsync(200, 0.0);

        expect(after, lessThan(before));
      },
      timeout: const Timeout(Duration(minutes: 2)),
    );
  });

  group('WebGPU training at scale (multi-workgroup dispatch)', () {
    test(
      'a larger multi-layer network trains on the GPU',
      () async {
        if (!onGpu()) return;

        // 12 inputs, 2 hidden layers and 3 outputs over 40 samples: every
        // dimension (weights, neurons, samples) is above the shader workgroup
        // sizes (64 / 8x8), so more than one workgroup is dispatched.
        final samples = randomSamples(40, 12, 3, seed: 71);
        final ann = ANN(
          scale,
          LayerFloat32x4(12, true, ActivationFunctionLinear()),
          [HiddenLayerConfig(24, true), HiddenLayerConfig(16, true)],
          LayerFloat32x4(3, false),
          random: Random(73),
        );

        final t = WebGpuRProp(ann, SamplesSet(samples, subject: 'random'))
          ..logEnabled = false;

        expect(await t.isWebGpuAccelerated, isTrue);

        final before = ann.computeSamplesGlobalError(samples);
        final after = await t.trainAsync(300, 0.0);

        expect(
          after,
          lessThan(before),
          reason: 'GPU training must reduce the global error',
        );

        // The GPU forward pass must still match the Dart one after training.
        for (final s in samples.take(4)) {
          ann.activate(s.input);
          final dartOut = ann.outputAsDouble;
          final gpuOut = await t.activateWebGpu(s.input);
          expect(maxAbsDiff(dartOut, gpuOut!), lessThan(1e-4));
        }
      },
      timeout: const Timeout(Duration(minutes: 5)),
    );

    test(
      'GPU and Dart agree on a larger network after a few epochs',
      () async {
        if (!onGpu()) return;

        final samples = randomSamples(24, 8, 2, seed: 83);

        ANNF larger() => ANN(
          scale,
          LayerFloat32x4(8, true, ActivationFunctionLinear()),
          [HiddenLayerConfig(16, true)],
          LayerFloat32x4(2, false),
          random: Random(89),
        );

        final dartTrainer = Backpropagation(
          larger(),
          SamplesSet(samples, subject: 'random'),
        )..logEnabled = false;
        final gpuTrainer = WebGpuBackpropagation(
          larger(),
          SamplesSet(samples, subject: 'random'),
        )..logEnabled = false;

        dartTrainer.train(5, 0.0);
        await gpuTrainer.trainAsync(5, 0.0);

        expect(
          maxAbsDiff(
            dartTrainer.ann.allWeights.cast<double>(),
            gpuTrainer.ann.allWeights.cast<double>(),
          ),
          lessThan(1e-3),
          reason: 'max |Δweight| after 5 epochs on a 8-16-2 network',
        );
      },
      timeout: const Timeout(Duration(minutes: 3)),
    );
  });

  group('WebGPU fallback in the browser', () {
    test(
      'an unsupported activation function falls back to pure Dart',
      () async {
        final samples = xor();
        final ann = ANN(
          scale,
          LayerFloat32x4(2, true, ActivationFunctionLinear()),
          [HiddenLayerConfig(4, true, _UnsupportedActivationFunction())],
          LayerFloat32x4(1, false),
          random: Random(97),
        );

        final t = WebGpuRProp(ann, SamplesSet(samples, subject: 'xor'))
          ..logEnabled = false;

        // No native/WebGPU activation id for this function -> no GPU backend.
        expect(await t.isWebGpuAccelerated, isFalse);

        // Still trains, on the synchronous pure-Dart path.
        final before = ann.computeSamplesGlobalError(samples);
        final after = await t.trainAsync(100, 0.0);
        expect(after, lessThan(before));
      },
    );
  });
}

/// A `Float32x4` activation function unknown to the accelerated backends
/// (`activationIdOf` returns `null` for its name), used to check that the
/// trainers fall back to pure Dart for unsupported networks.
class _UnsupportedActivationFunction extends ActivationFunctionFloat32x4 {
  _UnsupportedActivationFunction() : super('TestUnsupported', 1.0);

  static final List<ActivationFunctionScope> _scope = List.unmodifiable([
    ActivationFunctionScope.any,
  ]);

  @override
  List<ActivationFunctionScope> get scope => _scope;

  @override
  double activate(double x) => x < 0 ? 0.0 : x;

  @override
  Float32x4 activateEntry(Float32x4 entry) =>
      entry.max(ActivationFunctionFloat32x4.entryOfZeroes);

  @override
  double derivative(double o) => o <= 0 ? 0.0 : 1.0;

  @override
  Float32x4 derivativeEntry(Float32x4 entry) => entry
      .greaterThan(ActivationFunctionFloat32x4.entryOfZeroes)
      .select(
        ActivationFunctionFloat32x4.entryOfOnes,
        ActivationFunctionFloat32x4.entryOfZeroes,
      );
}
