import 'dart:typed_data';

/// A resident WebGPU trainer for a single network (weights, optimizer state and
/// the training sample set live in GPU buffers).
///
/// WebGPU is asynchronous (buffer readback requires `mapAsync`), so — unlike the
/// synchronous native [NativeAccelerator] — every operation that observes GPU
/// results returns a [Future].
abstract class WebGpuAccelerator {
  /// Total number of weights (flat `ANN.allWeights` order).
  int get weightsLength;

  /// Uploads weights (flat `ANN.allWeights` order) to GPU memory.
  Future<void> setWeights(Float32List weights);

  /// Downloads weights (flat `ANN.allWeights` order) from GPU memory.
  Future<Float32List> getWeights();

  /// Uploads the resident training sample set.
  Future<void> setSamples(
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
  Future<void> resetOptimizer();

  /// Runs one full training epoch over all resident samples and returns the
  /// epoch's global learn error. `lr`/`momentum` are used only by the
  /// Backpropagation rule. If the error is below [target], weights are NOT
  /// updated (mirrors `Propagation.learn`).
  Future<double> runEpoch(double target, double lr, double momentum);

  /// Runs a forward pass (inference) for a single [input], returning the output
  /// layer values.
  Future<Float32List> activate(Float32List input, int outputSize);

  /// Releases GPU resources.
  void dispose();
}
