import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

import 'eneural_net_training_propagation.dart';

/// Builds a learning-rate [ParameterStrategy] for an optimizer (used to plug in
/// LR schedules). Receives the optimizer (a [Propagation]) and its base LR.
typedef LearningRateScheduleBuilder<
  N extends num,
  E,
  T extends Signal<N, E, T>
> =
    ParameterStrategy<N, E, T> Function(
      Propagation<N, E, T, dynamic, dynamic> propagation,
      double baseLearningRate,
    );

/// Base class for per-weight gradient optimizers (Adam, RMSProp, AdaGrad,
/// AdaDelta, Quickprop, Lion, SGD/Momentum, ...).
///
/// It reuses the whole [Propagation] forward/backprop/batch-gradient machinery
/// and exposes a richer per-weight-entry update seam, [updateWeightEntry], that
/// (unlike [Propagation.computeWeightUpdate]) receives the layer/neuron/entry
/// coordinates — so an optimizer can index its own persistent state buffers
/// (allocated via [Propagation.createWeightStateBuffers]).
///
/// The base learning rate is exposed through the [ParameterStrategy] machine
/// (via [learningRate]), so learning-rate schedules can be plugged in by
/// overriding [createLearningRateStrategy]; the momentum strategy is static
/// (optimizers manage their own momentum state internally).
abstract class GradientOptimizer<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends Propagation<N, E, T, S, P> {
  /// The base (initial) learning rate.
  final double baseLearningRate;

  /// Mini-batch size. `0` (default) = full-batch (one weight update per epoch);
  /// `1` = online/pure SGD; otherwise samples are shuffled each epoch and the
  /// weights are updated once per mini-batch.
  final int batchSize;

  /// Decoupled L2 weight decay coefficient (AdamW-style); 0 disables it.
  /// Applied uniformly after the per-optimizer update: `Δw −= lr·wd·w`.
  final double weightDecay;

  /// Per-value gradient clipping bound; 0 disables it. Each gradient entry is
  /// clamped to `[-gradientClip, gradientClip]` before the update.
  final double gradientClip;

  /// Optional learning-rate schedule (e.g. [StepDecayStrategy],
  /// [CosineAnnealingStrategy]); when null a static base learning rate is used.
  final LearningRateScheduleBuilder<N, E, T>? lrSchedule;

  GradientOptimizer(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    required String algorithmName,
    this.baseLearningRate = 0.01,
    this.batchSize = 0,
    this.weightDecay = 0.0,
    this.gradientClip = 0.0,
    this.lrSchedule,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: algorithmName,
         subject: subject ?? samplesSet.subject,
       );

  late final E _clipLower = signalInstance.createEntryFullOf(
    signalInstance.toN(-gradientClip),
  );
  late final E _clipUpper = signalInstance.createEntryFullOf(
    signalInstance.toN(gradientClip),
  );

  /// The number of weight-update steps applied so far (one per full-batch epoch,
  /// or one per mini-batch). Used e.g. for Adam bias correction; reset by
  /// [reset] and settable when resuming from a checkpoint.
  int optimizerStep = 0;

  @override
  void reset() {
    super.reset();
    optimizerStep = 0;
  }

  @override
  ParameterStrategy<N, E, T> createLearningRateStrategy() => lrSchedule != null
      ? lrSchedule!(this, baseLearningRate)
      : StaticParameterStrategy(this, baseLearningRate);

  @override
  ParameterStrategy<N, E, T> createMomentumStrategy() =>
      StaticParameterStrategy(this, 0.0);

  /// The per-weight-entry update rule (SIMD). Returns the delta to ADD to the
  /// weight entry. Optimizers index their own state buffers with
  /// ([layerIndex], [neuronIndex], [entryIndex]).
  E updateWeightEntry({
    required int layerIndex,
    required int neuronIndex,
    required int entryIndex,
    required E weight,
    required E gradient,
    required E previousGradient,
    required E neuronOutput,
  });

  /// Mini-batch epoch: shuffles the samples and updates the weights once per
  /// mini-batch (reusing the protected backprop helpers). Falls back to the
  /// full-batch [Propagation.learn] when [batchSize] is 0 or covers all samples.
  @override
  bool learn(List<P> samples, double targetGlobalError) {
    if (batchSize <= 0 || batchSize >= samples.length) {
      return super.learn(samples, targetGlobalError);
    }

    final allLayers = ann.allLayers;
    final lastIndex = allLayers.length - 1;
    final lastLayer = allLayers[lastIndex];

    final order = List<int>.generate(samples.length, (i) => i)..shuffle(random);

    ann.trainingMode = true;
    for (var start = 0; start < order.length; start += batchSize) {
      final end = min(start + batchSize, order.length);
      ann.resetGradients();
      for (var k = start; k < end; ++k) {
        final sample = samples[order[k]];
        ann.activate(sample.input);
        backPropagateLastLayerError(lastLayer, lastIndex, sample.output);
        for (var i = allLayers.length - 2; i >= 0; --i) {
          backPropagateMiddleLayerError(allLayers[i], i);
        }
      }
      for (var i = 0; i < allLayers.length; ++i) {
        updateLayerWeights(allLayers[i], i);
      }
    }
    ann.trainingMode = false;

    final globalError = ann.computeSamplesGlobalError(samples);
    updateGlobalLearnError(globalError);
    return globalError < targetGlobalError;
  }

  @override
  void updateLayerWeights(Layer<N, E, T, S> layer, int layerIndex) {
    var nextLayer = layer.nextLayer;
    if (nextLayer == null) return;

    // The input layer (index 0) is always updated first in each weight-update
    // pass, so increment the step counter here (once per epoch / mini-batch).
    if (layerIndex == 0) ++optimizerStep;

    var neurons = layer.neurons;
    var length = neurons.length;

    var weights = layer.weights;
    var gradients = layer.gradients;
    var previousGradients = layer.previousGradients;

    var nextEntriesLength = nextLayer.neurons.valuesEntriesLength;

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronOutput = neurons.getValue(neuronI);
      var neuronWeights = weights[neuronI];
      var neuronGradients = gradients[neuronI];
      var neuronPreviousGradients = previousGradients[neuronI];

      var neuronOutputEntry = neurons.createEntryFullOf(neuronOutput);

      for (var i = 0; i < nextEntriesLength; ++i) {
        var weightsEntry = neuronWeights.getEntry(i);
        var gradientEntry = neuronGradients.getEntry(i);
        var previousGradientEntry = neuronPreviousGradients.getEntry(i);

        if (gradientClip > 0) {
          gradientEntry = signalInstance.entryOperationClamp(
            gradientEntry,
            _clipLower,
            _clipUpper,
          );
        }

        var delta = updateWeightEntry(
          layerIndex: layerIndex,
          neuronIndex: neuronI,
          entryIndex: i,
          weight: weightsEntry,
          gradient: gradientEntry,
          previousGradient: previousGradientEntry,
          neuronOutput: neuronOutputEntry,
        );

        if (weightDecay > 0) {
          delta = signalInstance.entryOperationSubtract(
            delta,
            signalInstance.entryOperationScale(
              weightsEntry,
              learningRate * weightDecay,
            ),
          );
        }

        var weight2 = neuronWeights.entryOperationSum(weightsEntry, delta);
        neuronWeights.setEntry(i, weight2);
      }
    }
  }

  /// Not used by [GradientOptimizer] subclasses (they override the richer
  /// [updateWeightEntry] seam); retained to satisfy the abstract
  /// [Propagation.computeWeightUpdate] contract.
  @override
  double computeWeightUpdate(
    N weight,
    N weightLastUpdate,
    num gradient,
    num previousGradient,
    List<num> previousUpdateDeltas,
    List<num> noImprovementCounter,
    int weightIndex,
    N neuronOutput,
  ) => throw UnsupportedError(
    '$algorithmName uses updateWeightEntry (not computeWeightUpdate)',
  );
}
