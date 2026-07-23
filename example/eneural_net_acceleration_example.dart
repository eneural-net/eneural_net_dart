import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

/// Example of native-accelerated training (macOS CPU / Metal).
///
/// `NativeRProp` and `NativeBackpropagation` are drop-in replacements for
/// `RProp`/`Backpropagation` on `Float32x4` networks. The whole training epoch
/// (forward + backprop + weight update over every sample) runs in native code
/// via Apple Accelerate (CPU) or Metal (GPU), reproducing the pure-Dart numerics
/// within float32 tolerance.
///
/// Build the native libraries first:
///   bash native/macos/build.sh
///
/// If no native library is available (other platforms, web, or a missing dylib)
/// the trainer transparently falls back to the pure-Dart SIMD path — this
/// example runs anywhere.
void main() {
  // Type of scale to use to compute the ANN:
  var scale = ScaleDouble.ZERO_TO_ONE;

  // The samples to learn in Float32x4 data type:
  var samples = SampleFloat32x4.toListFromString(
    ['0,0=0', '1,0=1', '0,1=1', '1,1=0'],
    scale,
    true, // Already normalized in the scale.
  );

  var samplesSet = SamplesSet(samples, subject: 'xor');

  // The activation function to use in the ANN:
  var activationFunction = ActivationFunctionSigmoid();

  // The ANN using layers that can compute with Float32x4 (SIMD compatible type).
  var ann = ANN(
    scale,
    // Input layer: 2 neurons with linear activation function:
    LayerFloat32x4(2, true, ActivationFunctionLinear()),
    // 1 Hidden layer: 3 neurons with sigmoid activation function:
    [HiddenLayerConfig(3, true, activationFunction)],
    // Output layer: 1 neuron with sigmoid activation function:
    LayerFloat32x4(1, false, activationFunction),
  );

  print(ann);

  // Native-accelerated iRProp+ trainer.
  //
  // `backend` can be:
  //   NativeBackend.auto  -> CPU when available, else Metal, else pure Dart (default)
  //   NativeBackend.cpu   -> Apple Accelerate (BLAS/vDSP)
  //   NativeBackend.metal -> Apple Metal (GPU)
  //   NativeBackend.none  -> force the pure-Dart SIMD path
  var trainer = NativeRProp(ann, samplesSet, backend: NativeBackend.auto);

  print(trainer);
  print('isNativeAccelerated: ${trainer.isNativeAccelerated}');
  print('activeBackend: ${trainer.activeBackend}');

  print('\n---------------------------------------------------');

  var chronometer = Chronometer('NativeRProp').start();

  // Train until global error 1e-4 (max 50000 epochs, up to 10 retries).
  var achievedTargetError = trainer.trainUntilGlobalError(
    targetGlobalError: 1.0e-4,
    maxEpochs: 50000,
    maxRetries: 10,
  );

  chronometer.stop(operations: trainer.totalTrainingActivations);

  print('---------------------------------------------------\n');

  // Compute the current global error of the ANN:
  var globalError = ann.computeSamplesGlobalError(samples);

  print('Samples Outputs:');
  for (var i = 0; i < samples.length; ++i) {
    var sample = samples[i];

    var input = sample.input;
    var expected = sample.output;

    // Activate the sample input (pure-Dart inference on the trained weights):
    ann.activate(input);
    var output = ann.output;

    // The same forward pass computed by the native backend (or null when the
    // trainer is running the pure-Dart fallback):
    var nativeOutput = trainer.activateNative(input);

    print(
      '- $i> $input -> $output ($expected) '
      '> error: ${output - expected} ; native: $nativeOutput',
    );
  }

  print('\nglobalError: $globalError');
  print('achievedTargetError: $achievedTargetError');
  print('trainedEpochs: ${trainer.totalTrainedEpochs}\n');

  print(chronometer);
}
