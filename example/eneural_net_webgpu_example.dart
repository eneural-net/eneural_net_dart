import 'package:eneural_net/eneural_net.dart';

/// Example of WebGPU-accelerated training.
///
/// `WebGpuRProp` / `WebGpuBackpropagation` add async training methods
/// (`trainUntilGlobalErrorAsync`, `trainAsync`) that run the whole epoch on the
/// browser GPU via WebGPU. WebGPU is asynchronous, hence the `Future` API.
///
/// This example runs anywhere: in a browser with WebGPU it trains on the GPU;
/// on the Dart VM (or a browser without WebGPU) it transparently falls back to
/// the pure-Dart SIMD trainer.
///
/// On the web, compile and serve it:
///   dart compile js example/eneural_net_webgpu_example.dart -o web/main.js
/// (with an HTML page loading `main.js`), or run it under `webdev`.
Future<void> main() async {
  // Type of scale to use to compute the ANN:
  var scale = ScaleDouble.ZERO_TO_ONE;

  // The samples to learn in Float32x4 data type:
  var samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '1,0=1', '0,1=1', '1,1=0'],
    scale,
    true, // Already normalized in the scale.
  );

  var samplesSet = SamplesSet(samples, subject: 'xor');

  var activationFunction = ActivationFunctionSigmoid();

  var ann = ANN(
    scale,
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    [HiddenLayerConfig(3, true, activationFunction)],
    LayerFloat32x4(1, false, activationFunction),
  );

  print(ann);

  // WebGPU-accelerated iRProp+ trainer (async).
  var trainer = WebGpuRProp(ann, samplesSet);
  trainer.logEnabled = false;

  final onGpu = await trainer.isWebGpuAccelerated;
  print('WebGPU accelerated: $onGpu');

  print('\n---------------------------------------------------');

  var chronometer = Chronometer('WebGpuRProp').start();

  // Async training on the GPU (or pure-Dart fallback).
  var achievedTargetError = await trainer.trainUntilGlobalErrorAsync(
    targetGlobalError: 1.0e-4,
    maxEpochs: 5000,
  );

  chronometer.stop(operations: trainer.totalTrainedEpochs);

  print('---------------------------------------------------\n');

  var globalError = ann.computeSamplesGlobalError(samples);

  print('Samples Outputs:');
  for (var i = 0; i < samples.length; ++i) {
    var sample = samples[i];
    var input = sample.input;
    var expected = sample.output;

    ann.activate(input);
    var output = ann.output;

    // The same forward pass computed on the GPU (or null on the fallback path):
    var gpuOutput = await trainer.activateWebGpu(input);

    print('- $i> $input -> $output ($expected) ; gpu: $gpuOutput');
  }

  print('\nglobalError: $globalError');
  print('achievedTargetError: $achievedTargetError');
  print('trainedEpochs: ${trainer.totalTrainedEpochs}\n');

  print(chronometer);
}
