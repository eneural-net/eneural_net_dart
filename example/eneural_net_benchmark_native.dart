import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;
typedef RPropF =
    RProp<double, Float32x4, SignalFloat32x4, Scale<double>, SampleFloat32x4>;

/// Benchmarks the pure-Dart SIMD trainer against the native CPU (Accelerate)
/// and Metal (GPU) whole-epoch trainers on a non-trivial network.
///
/// Native backends require the dylibs to be built first:
///   bash native/macos/build.sh
///
/// Run with: `dart run example/eneural_net_benchmark_native.dart`
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  const inputSize = 64;
  const hiddenSize = 256;
  const outputSize = 16;
  const numSamples = 64;
  const epochs = 200;

  final rnd = Random(12345);

  // Synthetic dataset.
  List<SampleFloat32x4> dataset() => List.generate(numSamples, (_) {
    final input = List<double>.generate(inputSize, (_) => rnd.nextDouble());
    final output = List<double>.generate(outputSize, (_) => rnd.nextDouble());
    return SampleFloat32x4.fromNormalized(input, output, scale);
  });

  final samples = dataset();

  ANNF build() => ANN(
    scale,
    LayerFloat32x4(inputSize, true),
    [HiddenLayerConfig(hiddenSize, true)],
    LayerFloat32x4(outputSize, false),
    random: Random(777),
  );

  double run(String label, RPropF trainer) {
    trainer.logEnabled = false;
    // Warm-up (build native network, compile shaders, etc.).
    trainer.train(1, 0.0);

    final sw = Stopwatch()..start();
    trainer.train(epochs, 0.0);
    sw.stop();

    final epochsPerSec = epochs / (sw.elapsedMicroseconds / 1e6);
    print(
      '${label.padRight(28)}  '
      '${sw.elapsedMilliseconds.toString().padLeft(6)} ms  '
      '${epochsPerSec.toStringAsFixed(1).padLeft(8)} epochs/s  '
      'error=${trainer.globalError.toStringAsExponential(3)}',
    );
    return epochsPerSec;
  }

  print(
    'Network: $inputSize+ -> [$hiddenSize+] -> $outputSize   '
    'samples=$numSamples  epochs=$epochs\n',
  );

  final dart = run(
    'Pure Dart (SIMD)',
    RProp(build(), SamplesSet(samples, subject: 'bench')),
  );

  final cpu = run(
    'Native CPU (Accelerate)',
    NativeRProp(
      build(),
      SamplesSet(samples, subject: 'bench'),
      backend: NativeBackend.cpu,
    ),
  );

  final metalTrainer = NativeRProp(
    build(),
    SamplesSet(samples, subject: 'bench'),
    backend: NativeBackend.metal,
  );
  final metalActive =
      metalTrainer.isNativeAccelerated &&
      metalTrainer.activeBackend == NativeBackend.metal;
  final metal = metalActive ? run('Native Metal (GPU)', metalTrainer) : 0.0;

  print('\nSpeedup vs pure Dart:');
  print('  CPU:   ${(cpu / dart).toStringAsFixed(2)}x');
  if (metalActive) {
    print('  Metal: ${(metal / dart).toStringAsFixed(2)}x');
  } else {
    print('  Metal: (not available)');
  }
}
