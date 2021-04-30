import 'package:eneural_net/eneural_net.dart';

abstract class Training<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>, P extends Sample<N, E, T, S>> {
  ANN<N, E, T, S> ann;

  Training(this.ann);

  void learn(P sample);

  double train(List<P> samples, int epochs) {
    var samplesLength = samples.length;

    for (var epoch = epochs - 1; epoch >= 0; --epoch) {
      for (var i = samplesLength - 1; i >= 0; --i) {
        var sample = samples[i];
        learn(sample);
      }
    }

    return ann.computeSamplesGlobalError(samples);
  }
}

class Backpropagation<
    N extends num,
    E,
    T extends Signal<N, E, T>,
    S extends Scale<N>,
    P extends Sample<N, E, T, S>> extends Training<N, E, T, S, P> {
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

    /*
    _layersDelta = ann.allLayers
        .map((l) => List<num>.generate(l.length, (i) => 0.0))
        .toList();

    _layersPreviousDelta = ann.allLayers
        .map((l) => List<num>.generate(l.length, (i) => 0.0))
        .toList();
     */
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

    /*
    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronWeights = weights[neuronI];
      var neuronError = 0.0;

      for (var nextNeuronI = 0; nextNeuronI < nextLength; ++nextNeuronI) {
        var weight = neuronWeights.getValue(nextNeuronI);
        var nextNeuronDelta = nextDeltas[nextNeuronI];
        neuronError += weight * nextNeuronDelta;
      }

      var neuronOutput = neurons.getValue(neuronI);

      deltas[neuronI] =
          neuronError * activationFunction.derivative(neuronOutput);
    }
     */
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

      var derivative = activationFunction.derivativeX4(neuronsEntry);
      var delta = expected.entryOperationMultiply(error, derivative);

      deltas.setEntry(i, delta);
    }

    /*
    var activationFunction = layer.activationFunction;

    var neurons = layer.neurons;
    var deltas = _layersDelta[layerIndex]; // layer.deltas;
    var length = neurons.length;

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronOutput = neurons.getValue(neuronI);
      var error = expected[neuronI] - neuronOutput;
      deltas[neuronI] = error * activationFunction.derivative(neuronOutput);
    }
     */
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

        var wUpdate = computeWeightUpdateX4(weightsEntry, nextDeltasEntry,
            nextDeltaPreviousEntry, neuronOutputEntry);

        var weight2 = neuronWeights.entryOperationSum(weightsEntry, wUpdate);

        neuronWeights.setEntry(i, weight2);

        nextLayerPreviousDeltas.setEntry(i, nextDeltasEntry);
      }

      //

      /*
      for (var nextNeuronI = 0; nextNeuronI < nextLength; ++nextNeuronI) {
        var weight = neuronWeights.getValue(nextNeuronI);
        var nextDelta = nextDeltas[nextNeuronI];
        var nextDeltaPrevious = nextLayerPreviousDeltas[nextNeuronI];

        var wUpdate = computeWeightUpdate(
            weight, nextDelta, nextDeltaPrevious, neuronOutput);

        var weight2 = weight + wUpdate;

        neuronWeights.setValue(nextNeuronI, neuronWeights.toN(weight2));

        nextLayerPreviousDeltas[nextNeuronI] = nextDelta;
      }
       */
    }
  }

  double computeWeightUpdate(N weight, num nextLayerDelta,
      num nextLayerPreviousDelta, N neuronOutput) {
    var wUpdate = learningRate * nextLayerDelta * neuronOutput;
    return wUpdate;
  }

  E computeWeightUpdateX4(
      E weight, E nextLayerDelta, E nextLayerPreviousDelta, E neuronOutput) {
    var deltaOutput =
        _signalInstance.entryOperationMultiply(nextLayerDelta, neuronOutput);
    var wUpdate =
        _signalInstance.entryOperationMultiply(_learningRateEntry, deltaOutput);
    return wUpdate;
  }
}
