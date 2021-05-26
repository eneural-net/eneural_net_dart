import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

import 'eneural_net_training_parameter_strategy.dart';

/// Base class for propagation training algorithms (similar to Backpropagation).
abstract class Propagation<
    N extends num,
    E,
    T extends Signal<N, E, T>,
    S extends Scale<N>,
    P extends Sample<N, E, T, S>> extends Training<N, E, T, S, P> {
  late final T _signalInstance;

  T get signalInstance => _signalInstance;
  late T _outputErrors;

  late final List<T> _layersGradientsDeltas;
  late final List<T> _layersPreviousGradientsDeltas;

  late final List<List<T>> _layersPreviousUpdateDelta;
  late final List<List<T>> _layersNoImprovementCounter;
  late final List<List<T>> _layersWeightsLastUpdates;

  late final ParameterStrategy<N, E, T> _learningRateStrategy;

  late final ParameterStrategy<N, E, T> _momentumStrategy;

  Propagation(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    String algorithmName = 'Propagation',
    String subject = '',
  }) : super(ann, samplesSet, algorithmName, subject: subject) {
    _outputErrors = ann.outputLayer.neurons.createInstanceOfSameLength();

    _outputErrors.setAllEntriesWithValue(_outputErrors.one);

    _layersGradientsDeltas = ann.allLayers
        .map((l) => l.neurons.createInstanceOfSameLength())
        .toList();

    _layersPreviousGradientsDeltas = ann.allLayers
        .map((l) => l.neurons.createInstanceOfSameLength())
        .toList();

    _layersPreviousUpdateDelta = ann.allLayers.map((l) {
      if (!l.hasNextLayer) return <T>[];
      return List.generate(
          l.weights.length,
          (i) => l.weights[i]
              .createInstanceOfSameLengthFullOfValue(l.neurons.toN(0.10)));
    }).toList();

    _layersNoImprovementCounter = ann.allLayers.map((l) {
      if (!l.hasNextLayer) return <T>[];
      return List.generate(
          l.weights.length, (i) => l.weights[i].createInstanceOfSameLength());
    }).toList();

    _layersWeightsLastUpdates = ann.allLayers.map((l) {
      if (!l.hasNextLayer) return <T>[];
      return List.generate(
          l.weights.length, (i) => l.weights[i].createInstanceOfSameLength());
    }).toList();

    _signalInstance = _layersGradientsDeltas.first.createInstance(1);

    _learningRateStrategy = createLearningRateStrategy();
    _momentumStrategy = createMomentumStrategy();

    _learningRateStrategy.initializeValue();
    _momentumStrategy.initializeValue();
  }

  ParameterStrategy<N, E, T> createLearningRateStrategy() =>
      LearningRateStrategy(this);

  ParameterStrategy<N, E, T> createMomentumStrategy() =>
      MomentumRateStrategy(this);

  late Random _random;

  Random get random => _random;

  double generateRandomValuePositive(double range) =>
      _random.nextDouble() * range;

  double generateRandomValue(double range) =>
      (_random.nextDouble() * (range * 2)) - range;

  double generateRandomWeightUpdate(
      double range, double min, double max, double multiplier) {
    var wUpdate = generateRandomValue(range);
    wUpdate = wUpdate * multiplier;
    if (wUpdate < 0) {
      wUpdate = wUpdate.clamp(-max, -min);
    } else {
      wUpdate = wUpdate.clamp(min, max);
    }
    return wUpdate;
  }

  double generateRandomWeightUpdateByFactor(double weight, double factor,
      {double zeroPoint = 0.01, double multiplier = 1.0}) {
    var weightUpdateRatio = zeroPoint + generateRandomValuePositive(factor);
    var w = weight.toDouble();
    var w2 = (w * weightUpdateRatio * multiplier).clamp(-1.0E20, 1.0E20);

    var weightUpdate = w2 - w;

    //print('factor: $factor ; zeroPoint: $zeroPoint ; multiplier: $multiplier ; weightUpdate: $weightUpdate ; weight: $w -> $w2');

    return weightUpdate;
  }

  double _noImprovementRatio = 1.0E-4;

  /// Minimal improvement ratio.
  double get noImprovementRatio => _noImprovementRatio;

  set noImprovementRatio(double value) {
    if (value < 0) value = -value;
    if (value < 1.0E-20) value = 1.0E-20;
    _noImprovementRatio = value;
  }

  int _noImprovementLimit = 10;

  /// Limit of epochs without improvements. Used to trigger strategies.
  int get noImprovementLimit => _noImprovementLimit;

  @override
  void initializeTraining() {
    super.initializeTraining();

    _random = Random(ann.allWeightsHashcode);
    _noImprovementLimit = 3;
  }

  @override
  void reset() {
    super.reset();

    for (var l in _layersGradientsDeltas) {
      l.setAllEntriesEmpty();
    }

    for (var l in _layersPreviousGradientsDeltas) {
      l.setAllEntriesEmpty();
    }

    for (var weightsUpdateDelta in _layersPreviousUpdateDelta) {
      for (var updateDeltas in weightsUpdateDelta) {
        updateDeltas.setAllEntriesWithValue(updateDeltas.toN(0.10));
      }
    }

    for (var weightsNoImprovements in _layersNoImprovementCounter) {
      for (var noImprovements in weightsNoImprovements) {
        noImprovements.setAllEntriesWithValue(noImprovements.zero);
      }
    }

    for (var l in _layersWeightsLastUpdates) {
      for (var w in l) {
        w.setAllEntriesEmpty();
      }
    }

    _learningRateStrategy.resetValue();
    _momentumStrategy.resetValue();
  }

  /// Returns the current learning rate of the Backpropagation.
  double get learningRate => _learningRateStrategy.value;

  E get learningRateEntry => _learningRateStrategy.valueEntry;

  void setLearningRate(double learningRate) =>
      _learningRateStrategy.setValue(learningRate);

  /// Returns the current momentum rate of the Backpropagation.
  double get momentum => _momentumStrategy.value;

  E get momentumEntry => _momentumStrategy.valueEntry;

  void setMomentum(double momentum) => _momentumStrategy.setValue(momentum);

  @override
  String get parameters =>
      'learningRate: $learningRate ; momentum: $momentum ; noImprovementLimit: $noImprovementLimit';

  @override
  void initializeParameters() {
    _learningRateStrategy.initializeValue();
    _momentumStrategy.initializeValue();
  }

  @override
  void updateParameters() {
    _learningRateStrategy.updateValue();
    _momentumStrategy.updateValue();
  }

  double _globalLearnError = 1.0;

  /// The global error while updating weights.
  double get globalLearnError => _globalLearnError;

  double _lastGlobalLearnError = 1.0;

  /// The previous global error while updating weights.
  double get lastGlobalLearnError => _lastGlobalLearnError;

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    var allLayers = ann.allLayers;
    var allLayersLength = allLayers.length;

    var lastIndex = allLayersLength - 1;
    var lastLayer = allLayers[lastIndex];

    ann.resetGradients();
    _resetGradientsDeltas();

    var allSamplesError = 0.0;

    var samplesLength = samples.length;
    for (var i = samplesLength - 1; i >= 0; --i) {
      var sample = samples[i];

      ann.activate(sample.input);
      var expected = sample.output;

      {
        _backPropagateLastLayerError(lastLayer, lastIndex, expected);
        var outputGlobalError = _outputErrors.computeSumSquares();
        allSamplesError += outputGlobalError;
      }

      for (var i = allLayersLength - 2; i >= 0; --i) {
        var layer = allLayers[i];
        _backPropagateMiddleLayerError(layer, i);
      }
    }

    var allSamplesOutputsSize = _outputErrors.length * samples.length;
    var globalLearnError = allSamplesError / allSamplesOutputsSize;

    _lastGlobalLearnError = _globalLearnError;
    _globalLearnError = globalLearnError;

    if (globalLearnError < targetGlobalError) {
      return true;
    }

    checkBestTrainingError(globalLearnError);

    for (var i = 0; i < allLayersLength; ++i) {
      var layer = allLayers[i];
      _updateLayerWeights(layer, i);
    }

    return false;
  }

  void _resetGradientsDeltas() {
    for (var i = 0; i < _layersGradientsDeltas.length; ++i) {
      var gradientsDeltas = _layersGradientsDeltas[i];
      var previousGradientsDeltas = _layersPreviousGradientsDeltas[i];

      previousGradientsDeltas.setAllEntriesWith(gradientsDeltas);
      gradientsDeltas.setAllEntriesEmpty();
    }
  }

  void _backPropagateMiddleLayerError(Layer<N, E, T, S> layer, int layerIndex) {
    var nextLayer = layer.nextLayer!;
    var activationFunction = layer.activationFunction;

    var neurons = layer.neurons;
    var length = neurons.length;

    var gradients = layer.gradients;

    var gradientsDeltas = _layersGradientsDeltas[layerIndex];

    var weights = layer.weights;

    var nextNeurons = nextLayer.neurons;
    var nextGradientsDeltas = _layersGradientsDeltas[layerIndex + 1];
    var nextEntriesLength = nextNeurons.entriesLength;
    var nextEntriesLastIndex = nextEntriesLength - 1;

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronOutput = neurons.getValue(neuronI);
      var neuronOutputEntry = neurons.createEntryFullOf(neuronOutput);

      var neuronWeights = weights[neuronI];
      var neuronGradients = gradients[neuronI];

      var neuronError = 0.0;

      for (var nextNeuronEntryI = nextEntriesLastIndex;
          nextNeuronEntryI >= 0;
          --nextNeuronEntryI) {
        var weightsEntry = neuronWeights.getEntry(nextNeuronEntryI);
        var nextNeuronGradientDeltaEntry =
            nextGradientsDeltas.getEntry(nextNeuronEntryI);

        var weightGradient = neuronWeights.entryOperationMultiply(
            neuronOutputEntry, nextNeuronGradientDeltaEntry);
        neuronGradients.addToEntry(nextNeuronEntryI, weightGradient);

        var entryErrors = neuronWeights.entryOperationMultiply(
            weightsEntry, nextNeuronGradientDeltaEntry);

        var laneError = nextNeuronEntryI == nextEntriesLastIndex
            ? neuronWeights.entryOperationSumLanePartial(
                entryErrors, neuronWeights.lastEntryLength)
            : neuronWeights.entryOperationSumLane(entryErrors);

        neuronError += laneError;
      }

      var derivative = layerIndex > 0
          ? activationFunction.derivativeWithFlatSpot(neuronOutput)
          : activationFunction.derivative(neuronOutput);

      var gradientDelta = neuronError * derivative;

      gradientsDeltas.setValue(neuronI, gradientsDeltas.toN(gradientDelta));
    }
  }

  void _backPropagateLastLayerError(
      Layer<N, E, T, S> layer, int layerIndex, T expected) {
    var activationFunction = layer.activationFunction;

    var neurons = layer.neurons;

    var gradientsDelta = _layersGradientsDeltas[layerIndex]; // layer.deltas;

    for (var i = neurons.entriesLength - 1; i >= 0; --i) {
      var neuronsEntry = neurons.getEntry(i);
      var expectedEntry = expected.getEntry(i);
      var error = expected.entryOperationSubtract(expectedEntry, neuronsEntry);

      _outputErrors.setEntry(i, error);

      var derivative =
          activationFunction.derivativeEntryWithFlatSpot(neuronsEntry);
      var gradientDelta = expected.entryOperationMultiply(error, derivative);

      gradientsDelta.setEntry(i, gradientDelta);
    }
  }

  void _updateLayerWeights(Layer<N, E, T, S> layer, int layerIndex) {
    var nextLayer = layer.nextLayer;
    if (nextLayer == null) return;

    var neurons = layer.neurons;
    var length = neurons.length;

    var weights = layer.weights;
    var gradients = layer.gradients;
    var previousGradients = layer.previousGradients;

    var weightsLastUpdates = _layersWeightsLastUpdates[layerIndex];

    var nextEntriesLength = nextLayer.neurons.entriesLength;

    var previousUpdateDeltas = _layersPreviousUpdateDelta[layerIndex];
    var noImprovementCounter = _layersNoImprovementCounter[layerIndex];

    for (var neuronI = 0; neuronI < length; ++neuronI) {
      var neuronOutput = neurons.getValue(neuronI);
      var neuronWeights = weights[neuronI];
      var neuronGradients = gradients[neuronI];
      var neuronPreviousGradients = previousGradients[neuronI];

      var neuronWeightsUpdates = weightsLastUpdates[neuronI];
      var weightsPreviousUpdateDeltas = previousUpdateDeltas[neuronI];
      var weightsNoImprovementCounter = noImprovementCounter[neuronI];

      var neuronOutputEntry = neurons.createEntryFullOf(neuronOutput);

      for (var i = 0; i < nextEntriesLength; ++i) {
        var weightsEntry = neuronWeights.getEntry(i);
        var weightsUpdatesEntry = neuronWeightsUpdates.getEntry(i);

        var nextGradientsEntry = neuronGradients.getEntry(i);
        var nextPreviousGradientsEntry = neuronPreviousGradients.getEntry(i);

        var wUpdate = computeEntryWeightUpdate(
            weightsEntry,
            weightsUpdatesEntry,
            nextGradientsEntry,
            nextPreviousGradientsEntry,
            weightsPreviousUpdateDeltas,
            weightsNoImprovementCounter,
            i,
            neuronOutputEntry);

        neuronWeightsUpdates.setEntry(i, wUpdate);

        var weight2 = neuronWeights.entryOperationSum(weightsEntry, wUpdate);

        neuronWeights.setEntry(i, weight2);
      }
    }
  }

  /// Implementation of the weight update.
  double computeWeightUpdate(
    N weight,
    N weightLastUpdate,
    num gradient,
    num previousGradient,
    List<num> previousUpdateDeltas,
    List<num> noImprovementCounter,
    int weightIndex,
    N neuronOutput,
  );

  /// Implementation of the weight update for an entry (SIMD).
  E computeEntryWeightUpdateSIMD(
    E weight,
    E weightLastUpdate,
    E gradient,
    E previousGradient,
    T previousUpdateDeltas,
    T noImprovementCounter,
    int weightsEntryIndex,
    E neuronOutput,
  ) {
    throw UnsupportedError('No SIMD implementation for: $this');
  }

  E computeEntryWeightUpdate(
    E weight,
    E weightLastUpdate,
    E gradient,
    E previousGradient,
    T previousUpdateDeltas,
    T noImprovementCounter,
    int weightsEntryIndex,
    E neuronOutput,
  ) {
    return _computeEntryWeightUpdateNonSIMD(
        weight,
        weightLastUpdate,
        gradient,
        previousGradient,
        previousUpdateDeltas,
        noImprovementCounter,
        weightsEntryIndex,
        neuronOutput);
  }

  E _computeEntryWeightUpdateNonSIMD(
    E weight,
    E weightLastUpdate,
    E gradient,
    E previousGradient,
    T previousUpdateDeltas,
    T noImprovementCounter,
    int weightsEntryIndex,
    E neuronOutput,
  ) {
    var entryBlockSize = _signalInstance.entryBlockSize;

    var weightsOffset = weightsEntryIndex * entryBlockSize;

    var listPreviousUpdateDeltas = List<N>.generate(entryBlockSize,
        (i) => previousUpdateDeltas.getValue(weightsOffset + i));

    var listNoImprovementCounter = List<N>.generate(entryBlockSize,
        (i) => noImprovementCounter.getValue(weightsOffset + i));

    var weightsUpdates = List<N>.generate(entryBlockSize, (i) {
      var valWeight = _signalInstance.getValueFromEntry(weight, i);
      var valWeightLastUp =
          _signalInstance.getValueFromEntry(weightLastUpdate, i);

      var valGradient = _signalInstance.getValueFromEntry(gradient, i);
      var valPrevGradient =
          _signalInstance.getValueFromEntry(previousGradient, i);

      var valNeuronOutput = _signalInstance.getValueFromEntry(neuronOutput, i);

      var weightUp = computeWeightUpdate(
        valWeight,
        valWeightLastUp,
        valGradient,
        valPrevGradient,
        listPreviousUpdateDeltas,
        listNoImprovementCounter,
        i,
        valNeuronOutput,
      );

      return _signalInstance.toN(weightUp);
    });

    var deltasEntry = _signalInstance.createEntry(listPreviousUpdateDeltas);
    previousUpdateDeltas.setEntry(weightsEntryIndex, deltasEntry);

    var noImprovementEntry =
        _signalInstance.createEntry(listNoImprovementCounter);
    noImprovementCounter.setEntry(weightsEntryIndex, noImprovementEntry);

    return _signalInstance.createEntry(weightsUpdates);
  }
}
