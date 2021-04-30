import 'dart:typed_data';

import 'eneural_net_activation_functions.dart';
import 'eneural_net_extension.dart';
import 'eneural_net_sample.dart';
import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';

class ANN<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>> {
  final S scale;

  final LayerInput<N, E, T, S> inputLayer;
  late final List<LayerHidden<N, E, T, S>> hiddenLayers;
  final LayerOutput<N, E, T, S> outputLayer;

  late final List<Layer<N, E, T, S>> allLayers;

  ANN(this.scale, Layer<N, E, T, S> inputLayer, List<int> hiddenLayersNeurons,
      Layer<N, E, T, S> outputLayer)
      : inputLayer = inputLayer.asLayerInput,
        outputLayer = outputLayer.asLayerOutput {
    _build(hiddenLayersNeurons);
  }

  void _build(List<int> hiddenLayersNeurons) {
    hiddenLayers = hiddenLayersNeurons
        .map((n) => LayerHidden<N, E, T, S>(
            inputLayer.neurons.createInstance(n),
            inputLayer.activationFunction))
        .toList();

    allLayers = [inputLayer, ...hiddenLayers, outputLayer];

    for (var i = allLayers.length - 2; i >= 0; --i) {
      var layer = allLayers[i];
      var layerNext = allLayers[i + 1];
      layer.connectTo(layerNext);
    }
  }

  void activate(T signal) {
    inputLayer.resetNextLayerNetwork();
    inputLayer.setNeurons(signal);

    inputLayer.activateLayer();
  }

  void resetWeights() {
    for (var i = allLayers.length - 1; i >= 0; --i) {
      var l = allLayers[i];
      l.resetWeights();
    }
  }

  List<N> get output => outputLayer.getNeurons();

  List<N> get outputDenormalized => outputLayer.getNeuronsDenormalized(scale);

  List<num> computeSamplesErrors<P extends Sample<N, E, T, S>>(
      List<P> samples) {
    return samples.map((s) {
      activate(s.input);
      var outputErrors = output.diffFromSignal(s.output);
      var outputErrorsSquareMeanRoot = outputErrors.square.mean.squareRoot;
      return outputErrorsSquareMeanRoot;
    }).toList();
  }

  double computeSamplesGlobalError<P extends Sample<N, E, T, S>>(
          List<P> samples) =>
      computeSamplesErrors(samples).sumSquares / samples.length;

  @override
  String toString() {
    return '$runtimeType{ '
        'layers: ${inputLayer.length} -> '
        '${hiddenLayers.map((l) => l.length).toList()} -> '
        '${outputLayer.length} ; '
        '$scale  ; ${inputLayer.activationFunction} }';
  }
}

class LayerInt32 extends Layer<int, Int32x4, SignalInt32, Scale<int>> {
  LayerInt32(int size,
      [ActivationFunction<int, Int32x4> activationFunction =
          const ActivationFunctionSigmoidFastInt100()])
      : super(SignalInt32(size), activationFunction);
}

class LayerFloat32
    extends Layer<double, Float32x4, SignalFloat32, Scale<double>> {
  LayerFloat32(int size,
      [ActivationFunction<double, Float32x4> activationFunction =
          const ActivationFunctionSigmoid()])
      : super(SignalFloat32(size), activationFunction);
}

class Layer<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>> {
  final T _neurons;
  final ActivationFunction<N, E> activationFunction;

  late List<T> _weights;

  List<T> get weights => _weights;

  Layer<N, E, T, S>? _previousLayer;
  Layer<N, E, T, S>? _nextLayer;

  Layer<N, E, T, S>? get nextLayer => _nextLayer;

  Layer<N, E, T, S>? get previousLayer => _previousLayer;

  Layer(this._neurons, this.activationFunction);

  T get neurons => _neurons;

  int get length => _neurons.length;

  void connectTo(Layer<N, E, T, S> nextLayer) {
    var inSize = _neurons.length;
    var outSize = nextLayer._neurons.length;

    _weights = List.generate(inSize, (index) {
      var weights = neurons.createRandomInstance(
          outSize, activationFunction.initialWeightScale);
      weights.setExtraValues(neurons.zero);
      return weights;
    });

    _nextLayer = nextLayer;
    nextLayer._previousLayer = this;
  }

  void resetWeights() {
    var initialWeightScale = activationFunction.initialWeightScale;

    for (var i = 0; i < _weights.length; ++i) {
      var weights = _weights[i];

      for (var j = 0; j < weights.entriesLength; ++j) {
        weights.setEntryWithRandomValues(j, initialWeightScale);
      }
    }
  }

  void setNeurons(T signal) {
    _neurons.set(signal);
  }

  List<N> getNeurons() {
    return _neurons.values;
  }

  List<N> getNeuronsDenormalized(S scale) =>
      scale.denormalizeList(getNeurons());

  void resetNetwork() {
    resetLayer();

    _nextLayer?.resetNetwork();
  }

  void resetNextLayerNetwork() {
    _nextLayer?.resetNetwork();
  }

  void resetLayer() {
    for (var i = _neurons.entriesLength - 1; i >= 0; --i) {
      _neurons.setEntryEmpty(i);
    }
  }

  void activateLayer() {
    throw UnsupportedError(
        'Should use a LayerInput, LayerHidden or LayerOutput instance!');
  }

  LayerInput<N, E, T, S> get asLayerInput =>
      LayerInput(neurons, activationFunction);

  LayerHidden<N, E, T, S> get asLayerHidden =>
      LayerHidden(neurons, activationFunction);

  LayerOutput<N, E, T, S> get asLayerOutput =>
      LayerOutput(neurons, activationFunction);

  @override
  String toString() {
    return 'Layer{output: ${_neurons.valuesAsString}';
  }
}

class LayerInput<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  late final Layer<N, E, T, S> _nextLayerNonNull;

  LayerInput(T neurons, ActivationFunction<N, E> activationFunction)
      : super(neurons, activationFunction);

  @override
  void connectTo(Layer<N, E, T, S> nextLayer) {
    super.connectTo(nextLayer);
    _nextLayerNonNull = _nextLayer!;
  }

  @override
  void activateLayer() {
    var nextLayer = _nextLayerNonNull;

    for (var i = _neurons.length - 1; i >= 0; --i) {
      var neuronOutput = _neurons.getValue(i);
      var weights = _weights[i];

      weights.multiplyValueAddingTo(neuronOutput, nextLayer._neurons);
    }

    nextLayer.activateLayer();
  }

  @override
  LayerInput<N, E, T, S> get asLayerInput => this;
}

class LayerHidden<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  late final Layer<N, E, T, S> _nextLayerNonNull;

  LayerHidden(T neurons, ActivationFunction<N, E> activationFunction)
      : super(neurons, activationFunction);

  @override
  void connectTo(Layer<N, E, T, S> nextLayer) {
    super.connectTo(nextLayer);
    _nextLayerNonNull = _nextLayer!;
  }

  @override
  void activateLayer() {
    var activationFunction = this.activationFunction.activateX4;

    for (var i = _neurons.entriesLength - 1; i >= 0; --i) {
      _neurons.setEntryFilteredX4(i, activationFunction);
    }

    var nextLayer = _nextLayerNonNull;

    for (var i = _neurons.length - 1; i >= 0; --i) {
      var neuronOutput = _neurons.getValue(i);
      var weights = _weights[i];

      weights.multiplyValueAddingTo(neuronOutput, nextLayer._neurons);
    }

    nextLayer.activateLayer();
  }

  @override
  LayerHidden<N, E, T, S> get asLayerHidden => this;
}

class LayerOutput<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  LayerOutput(T neurons, ActivationFunction<N, E> activationFunction)
      : super(neurons, activationFunction);

  @override
  void activateLayer() {
    var activationFunction = this.activationFunction.activateX4;

    for (var i = _neurons.entriesLength - 1; i >= 0; --i) {
      _neurons.setEntryFilteredX4(i, activationFunction);
    }
  }

  @override
  LayerOutput<N, E, T, S> get asLayerOutput => this;
}
