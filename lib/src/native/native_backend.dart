import 'dart:typed_data';

/// Native acceleration backend selection.
enum NativeBackend {
  /// No native acceleration; use the pure-Dart SIMD path.
  none,

  /// Apple Accelerate (BLAS/vDSP) CPU backend.
  cpu,

  /// Apple Metal (GPU) backend.
  metal,

  /// NVIDIA CUDA (GPU) backend (Windows/Linux).
  cuda,

  /// Automatically pick the best available backend for the host:
  /// on macOS the Metal (GPU) backend when worthwhile else CPU (Accelerate);
  /// on Windows/Linux the CUDA (GPU) backend when available; otherwise [none].
  auto,
}

/// Immutable description of an [ANN] topology used to build a native network.
///
/// This is a plain-data value (no `dart:ffi`/`dart:io`) so it is safe to
/// reference from web builds.
class NativeNetworkSpec {
  /// Neuron count of each layer, INCLUDING the bias slot when present.
  final List<int> neuronCounts;

  /// Whether each layer has a bias neuron.
  final List<bool> withBias;

  /// Activation function id of each layer:
  /// 0=Linear, 1=Sigmoid, 2=SigmoidFast, 3=SigmoidBoundedFast.
  final List<int> activationIds;

  /// Per-layer activation scale (used by `SigmoidBoundedFast`).
  final List<double> activationScale;

  /// Per-layer activation flat-spot.
  final List<double> flatSpots;

  const NativeNetworkSpec({
    required this.neuronCounts,
    required this.withBias,
    required this.activationIds,
    required this.activationScale,
    required this.flatSpots,
  });

  int get numLayers => neuronCounts.length;
}

/// A resident native trainer for a single network (weights, optimizer state and
/// the training sample set live in native memory).
///
/// Implementations wrap a native library handle; see the `io` implementation.
/// On unsupported platforms (web) resolution simply returns `null`.
abstract class NativeAccelerator {
  /// The backend actually in use.
  NativeBackend get backend;

  /// Total number of weights (flat `ANN.allWeights` order).
  int get weightsLength;

  /// Uploads weights (flat `ANN.allWeights` order) to native memory.
  void setWeights(Float32List weights);

  /// Downloads weights (flat `ANN.allWeights` order) from native memory.
  void getWeights(Float32List out);

  /// Uploads the resident training sample set.
  void setSamples(
    int numSamples,
    int inputSize,
    int outputSize,
    Float32List inputs,
    Float32List targets,
  );

  /// Selects the Backpropagation update rule.
  void configBackprop();

  /// Selects the iRProp+ update rule.
  void configRProp(
    double delta0,
    double deltaMin,
    double deltaMax,
    double etaPlus,
    double etaMinus,
  );

  /// Resets optimizer state for a fresh training session.
  void resetOptimizer();

  /// Runs one full training epoch over all resident samples and returns the
  /// epoch's global learn error. `lr`/`momentum` are used only by the
  /// Backpropagation rule. If the error is below [target], weights are NOT
  /// updated (mirrors `Propagation.learn`).
  double runEpoch(double target, double lr, double momentum);

  /// Runs a forward pass (inference) for a single [input] into [output].
  void activate(Float32List input, Float32List output);

  /// Releases native resources.
  void dispose();
}
