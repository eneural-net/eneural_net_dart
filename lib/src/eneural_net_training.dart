import 'package:eneural_net/eneural_net.dart';

import 'eneural_net_extension.dart';

/// Base class for training algorithms.
abstract class Training<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>, P extends Sample<N, E, T, S>> {
  ANN<N, E, T, S> ann;

  Training(this.ann);

  /// Learn the training of [sample]. Called by [train].
  void learn(P sample);

  /// Reset this instance for a future training sessions.
  void reset() {
    _lastGlobalError = double.maxFinite;
    _trainedEpochs = 0;
    _trainingActivations = 0;
  }

  /// Train the [ann] until [targetGlobalError],
  /// with [maxEpochs] per training session and
  /// a [maxRetries] when a training session can't reach the target global error.
  bool trainUntilGlobalError(List<P> samples,
      {double targetGlobalError = 0.01,
      int maxEpochs = 1000000,
      int maxRetries = 5}) {
    for (var retry = 0; retry <= maxRetries; ++retry) {
      reset();
      ann.resetWeights();

      while (
          _lastGlobalError > targetGlobalError && _trainedEpochs < maxEpochs) {
        train(samples, 100);
      }
      if (_lastGlobalError <= targetGlobalError) {
        return true;
      }
    }
    return false;
  }

  int _totalTrainedEpochs = 0;

  /// Returns the total number of epochs of all the training session.
  /// A call to [reset] won't reset this value.
  int get totalTrainedEpochs => _totalTrainedEpochs;

  int _trainedEpochs = 0;

  /// Returns the number of epochs of the last training session.
  int get trainedEpochs => _trainedEpochs;

  int _totalTrainingActivations = 0;

  /// Returns the total number of activations of all the training session.
  /// A call to [reset] won't reset this value.
  int get totalTrainingActivations => _totalTrainingActivations;

  int _trainingActivations = 0;

  /// Returns the number of activations of the last training session.
  int get trainingActivations => _trainingActivations;

  double _lastGlobalError = double.maxFinite;

  /// Returns the last training global error (set by [train]).
  double get lastGlobalError => _lastGlobalError;

  /// Train the [samples] for n [epochs] and returns the last
  /// global error.
  double train(List<P> samples, int epochs) {
    var samplesLength = samples.length;

    for (var epoch = epochs - 1; epoch >= 0; --epoch) {
      for (var i = samplesLength - 1; i >= 0; --i) {
        var sample = samples[i];
        learn(sample);
      }
    }

    _trainedEpochs += epochs;
    _totalTrainedEpochs += epochs;

    var activations = epochs * samplesLength;
    _trainingActivations += activations;
    _totalTrainingActivations += activations;

    var globalError = ann.computeSamplesGlobalError(samples);

    _lastGlobalError = globalError;

    return globalError;
  }
}

/// Implementation of Backpropagation training algorithm.
class Backpropagation<
    N extends num,
    E,
    T extends Signal<N, E, T>,
    S extends Scale<N>,
    P extends Sample<N, E, T, S>> extends Training<N, E, T, S, P> {
  /// The learning rate of the Backpropagation.
  final double learningRate;
  late final E _learningRateEntry;

  late final T _signalInstance;
  late final List<T> _layersDelta;
  late final List<T> _layersPreviousDelta;

  Backpropagation(ANN<N, E, T, S> ann, {this.learningRate = 0.20})
      : super(ann) {
    _layersDelta =
        ann.allLayers.map((l) => l.neurons.createInstance(l.length)).toList();

    _layersPreviousDelta =
        ann.allLayers.map((l) => l.neurons.createInstance(l.length)).toList();

    _signalInstance = _layersDelta.first.createInstance(1);

    _learningRateEntry =
        _signalInstance.createEntryFullOf(_signalInstance.toN(learningRate));
  }

  @override
  void reset() {
    super.reset();

    var fResetDelta = (i, v) {
      var l = ann.allLayers[i];
      return ann.allLayers[i].neurons.createInstance(l.length);
    };

    _layersDelta.setAllWith(fResetDelta);
    _layersPreviousDelta.setAllWith(fResetDelta);
  }

  @override
  void learn(P sample) {
    ann.activate(sample.input);

    var expected = sample.output;

    var allLayers = ann.allLayers;

    var allLayersLength = allLayers.length;

    {
      var lastIndex = allLayersLength - 1;
      var lastLayer = allLayers[lastIndex];
      _backPropagateLastLayerError(lastLayer, lastIndex, expected);
    }

    for (var i = allLayersLength - 2; i >= 0; --i) {
      var layer = allLayers[i];
      _backPropagateMiddleLayerError(layer, i);
    }

    for (var i = 0; i < allLayersLength; ++i) {
      var layer = allLayers[i];
      _updateLayerWeights(layer, i);
    }
  }

  void _backPropagateMiddleLayerError(Layer<N, E, T, S> layer, int layerIndex) {
    var nextLayer = layer.nextLayer!;
    var activationFunction = layer.activationFunction;

    var neurons = layer.neurons;
    var deltas = _layersDelta[layerIndex];
    var length = neurons.length;

    var weights = layer.weights;

    var nextNeurons = nextLayer.neurons;
    var nextDeltas = _layersDelta[layerIndex + 1];
    var nextEntriesLength = nextNeurons.entriesLength;

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronWeights = weights[neuronI];
      var neuronError = 0.0;

      for (var nextNeuronEntryI = nextEntriesLength - 1;
          nextNeuronEntryI >= 0;
          --nextNeuronEntryI) {
        var weightsEntry = neuronWeights.getEntry(nextNeuronEntryI);
        var nextNeuronDeltaEntry = nextDeltas.getEntry(nextNeuronEntryI);

        var entryErrors = neuronWeights.entryOperationMultiply(
            weightsEntry, nextNeuronDeltaEntry);
        neuronError += neuronWeights.entryOperationSumLane(entryErrors);
      }

      var neuronOutput = neurons.getValue(neuronI);

      var delta = neuronError * activationFunction.derivative(neuronOutput);

      deltas.setValue(neuronI, deltas.toN(delta));
    }
  }

  void _backPropagateLastLayerError(
      Layer<N, E, T, S> layer, int layerIndex, T expected) {
    var activationFunction = layer.activationFunction;

    var neurons = layer.neurons;
    var deltas = _layersDelta[layerIndex]; // layer.deltas;

    for (var i = neurons.entriesLength - 1; i >= 0; --i) {
      var neuronsEntry = neurons.getEntry(i);
      var expectedEntry = expected.getEntry(i);
      var error = expected.entryOperationSubtract(expectedEntry, neuronsEntry);

      var derivative = activationFunction.derivativeEntry(neuronsEntry);
      var delta = expected.entryOperationMultiply(error, derivative);

      deltas.setEntry(i, delta);
    }
  }

  void _updateLayerWeights(Layer<N, E, T, S> layer, int layerIndex) {
    var nextLayer = layer.nextLayer;
    if (nextLayer == null) return;

    var neurons = layer.neurons;
    var weights = layer.weights;
    var length = neurons.length;

    var nextLayerIndex = layerIndex + 1;
    var nextEntriesLength = nextLayer.neurons.entriesLength;
    var nextDeltas = _layersDelta[nextLayerIndex];

    var nextLayerPreviousDeltas = _layersPreviousDelta[nextLayerIndex];

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronOutput = neurons.getValue(neuronI);
      var neuronWeights = weights[neuronI];

      var neuronOutputEntry = neurons.createEntryFullOf(neuronOutput);

      for (var i = 0; i < nextEntriesLength; ++i) {
        var weightsEntry = neuronWeights.getEntry(i);
        var nextDeltasEntry = nextDeltas.getEntry(i);
        var nextDeltaPreviousEntry = nextLayerPreviousDeltas.getEntry(i);

        var wUpdate = computeEntryWeightUpdate(weightsEntry, nextDeltasEntry,
            nextDeltaPreviousEntry, neuronOutputEntry);

        var weight2 = neuronWeights.entryOperationSum(weightsEntry, wUpdate);

        neuronWeights.setEntry(i, weight2);

        nextLayerPreviousDeltas.setEntry(i, nextDeltasEntry);
      }
    }
  }

  /// Implementation of the weight update.
  double computeWeightUpdate(N weight, num nextLayerDelta,
      num nextLayerPreviousDelta, N neuronOutput) {
    var wUpdate = learningRate * nextLayerDelta * neuronOutput;
    return wUpdate;
  }

  /// Implementation of the weight update for an entry.
  E computeEntryWeightUpdate(
      E weight, E nextLayerDelta, E nextLayerPreviousDelta, E neuronOutput) {
    var deltaOutput =
        _signalInstance.entryOperationMultiply(nextLayerDelta, neuronOutput);
    var wUpdate =
        _signalInstance.entryOperationMultiply(_learningRateEntry, deltaOutput);
    return wUpdate;
  }
}
