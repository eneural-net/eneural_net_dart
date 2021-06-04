import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:swiss_knife/swiss_knife.dart' show encodeJSON, parseJSON;

import 'eneural_net_activation_functions.dart';
import 'eneural_net_extension.dart';
import 'eneural_net_sample.dart';
import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';

/// Artificial Neural Network
class ANN<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>> {
  final S scale;

  final LayerInput<N, E, T, S> inputLayer;
  late final List<LayerHidden<N, E, T, S>> hiddenLayers;
  final LayerOutput<N, E, T, S> outputLayer;

  late final List<Layer<N, E, T, S>> allLayers;

  ANN(this.scale, Layer<N, E, T, S> inputLayer,
      List<HiddenLayerConfig> hiddenLayersConfig, Layer<N, E, T, S> outputLayer,
      {Random? random})
      : inputLayer = inputLayer.asLayerInput,
        outputLayer = outputLayer.asLayerOutput {
    _build(hiddenLayersConfig, null, random);
  }

  ANN._(this.scale, Layer<N, E, T, S> inputLayer,
      Iterable<Layer<N, E, T, S>> hiddenLayers, Layer<N, E, T, S> outputLayer,
      {Random? random})
      : inputLayer = inputLayer.asLayerInput,
        outputLayer = outputLayer.asLayerOutput {
    _build(null, hiddenLayers, random);
  }

  void _build(
      [List<HiddenLayerConfig>? hiddenLayersConfig,
      Iterable<Layer<N, E, T, S>>? hiddenLayers,
      Random? random]) {
    if (outputLayer.withBiasNeuron) {
      throw StateError("Can't have bias neuron at output layer: $outputLayer");
    }

    if (hiddenLayersConfig != null) {
      var defaultActivationFunction = _defaultActivationFunction();

      this.hiddenLayers = hiddenLayersConfig.map((l) {
        return LayerHidden<N, E, T, S>._(
            inputLayer.neurons
                .createInstance((l.withBiasNeuron ? l.neurons + 1 : l.neurons)),
            l.withBiasNeuron,
            l.getActivationFunction(defaultActivationFunction));
      }).toList();
    } else {
      this.hiddenLayers = hiddenLayers!.map((e) => e.asLayerHidden).toList();
    }

    allLayers = [inputLayer, ...this.hiddenLayers, outputLayer];

    for (var i = allLayers.length - 2; i >= 0; --i) {
      var layer = allLayers[i];
      var layerNext = allLayers[i + 1];
      layer.connectTo(layerNext, random: random);
    }
  }

  ActivationFunction<N, E> _defaultActivationFunction() {
    var functions = [
      outputLayer.activationFunction,
      inputLayer.activationFunction,
    ];

    var defaultActivationFunction = functions.firstWhere(
      (f) => f.scope.contains(ActivationFunctionScope.input),
      orElse: () => outputLayer.activationFunction,
    );

    return defaultActivationFunction;
  }

  /// The signal format of this ANN.
  String get format => inputLayer.format;

  /// Size of the input layer (number of input neurons).
  int get inputSize => inputLayer.length;

  /// Size of the output layer (number of output neurons).
  int get outputSize => outputLayer.length;

  /// Returns a List with the number of neurons of each layer.
  List<int> get allLayersNeuronsSize => allLayers.map((e) => e.length).toList();

  /// Total number of weights of all layers.
  int get allWeightsLength {
    var all = 0;

    for (var i = hiddenLayers.length - 1; i >= 0; --i) {
      var l = hiddenLayers[i];

      for (var ws in l.weights) {
        all += ws.length;
      }
    }

    for (var ws in inputLayer.weights) {
      all += ws.length;
    }

    return all;
  }

  List<N> get allWeights {
    var allWeights = <N>[];

    for (var i = hiddenLayers.length - 1; i >= 0; --i) {
      var l = hiddenLayers[i];

      for (var ws in l.weights) {
        allWeights.addAll(ws.values);
      }
    }

    for (var ws in inputLayer.weights) {
      allWeights.addAll(ws.values);
    }

    return allWeights;
  }

  set allWeights(List<N> weights) {
    var weightsOffset = 0;

    for (var i = hiddenLayers.length - 1; i >= 0; --i) {
      var l = hiddenLayers[i];

      for (var ws in l.weights) {
        weightsOffset += ws.setAllWithList(weights, weightsOffset);
        ws.setExtraValuesToZero();
      }
    }

    for (var ws in inputLayer.weights) {
      weightsOffset += ws.setAllWithList(weights, weightsOffset);
      ws.setExtraValuesToZero();
    }
  }

  int get allWeightsHashcode => ListEquality<N>().hash(allWeights);

  /// Activate with [signal].
  void activate(T signal) {
    inputLayer.resetNextLayerNetwork();
    inputLayer.setNeurons(signal);

    inputLayer.activateLayer();
  }

  /// Reset the weights with random values.
  void resetWeights([Random? random]) {
    for (var i = allLayers.length - 1; i >= 0; --i) {
      var l = allLayers[i];
      l.resetWeights(random);
    }
  }

  /// Reset the gradients with zeroes.
  void resetGradients() {
    for (var i = allLayers.length - 1; i >= 0; --i) {
      var l = allLayers[i];
      l.resetGradients();
    }
  }

  /// Returns the current output of the [outputLayer] neurons as [List<N>].
  List<N> get output => outputLayer.getNeurons();

  /// Returns the current output of the [outputLayer] neurons as [List<double>].
  List<double> get outputAsDouble => outputLayer.getNeuronsAsDouble();

  List<N> get outputDenormalized => outputLayer.getNeuronsDenormalized(scale);

  List<List<double>> computeSamplesActivations<P extends Sample<N, E, T, S>>(
      List<P> samples) {
    return samples.map((s) {
      activate(s.input);
      return outputAsDouble;
    }).toList();
  }

  /// Computes the output errors for each sample in [samples].
  List<double> computeSamplesErrors<P extends Sample<N, E, T, S>>(
      List<P> samples) {
    return samples.map((s) {
      activate(s.input);
      var outputErrors = outputAsDouble.diffFromSignal(s.output);
      var outputErrorsSquareMeanRoot = outputErrors.squaresMean;
      return outputErrorsSquareMeanRoot;
    }).toList();
  }

  /// Computes the global error for [samples].
  double computeSamplesGlobalError<P extends Sample<N, E, T, S>>(
          List<P> samples) =>
      computeSamplesErrors(samples).mean;

  @override
  String toString() {
    var inputStr = inputLayer.withBiasNeuron
        ? '${inputLayer.length - 1}+'
        : '${inputLayer.length}';

    var hiddenStr = hiddenLayers
        .map((l) => l.withBiasNeuron ? '${l.length - 1}+' : '${l.length}')
        .toList()
        .toString();

    return '$runtimeType{ '
        'layers: $inputStr -> $hiddenStr -> ${outputLayer.length} ; '
        '$scale  ; ${hiddenLayers.first.activationFunction} }';
  }

  /// Converts this ANN to an encoded JSON.
  String toJson({bool withIndent = true}) =>
      encodeJSON(toJsonMap(), withIndent: withIndent);

  /// Converts this ANN to a JSON [Map].
  Map<String, dynamic> toJsonMap() => <String, dynamic>{
        'format': format,
        'scale': scale.toJsonMap(),
        'layers': allLayers.map((e) => e.toJsonMap()).toList()
      };

  static ANN fromJson(dynamic json) {
    Map<String, dynamic> jsonMap = json is String ? parseJSON(json) : json;

    var format = jsonMap['format'] as String;
    var scale = Scale.fromJson(jsonMap['scale']);
    var allLayers = (jsonMap['layers'] as List).cast<Map<String, dynamic>>();

    var inputLayer = allLayers.first;
    var hiddenLayers = allLayers.sublist(1, allLayers.length - 1);
    var outputLayer = allLayers.last;

    switch (format) {
      case 'Float32x4':
        {
          return ANN<double, Float32x4, SignalFloat32x4, Scale<double>>._(
            scale as Scale<double>,
            Layer<double, Float32x4, SignalFloat32x4, Scale<double>>.fromJson(
                inputLayer),
            hiddenLayers.map((e) => Layer<double, Float32x4, SignalFloat32x4,
                Scale<double>>.fromJson(e)),
            Layer<double, Float32x4, SignalFloat32x4, Scale<double>>.fromJson(
                outputLayer),
          );
        }
      case 'Int32x4':
        {
          return ANN<int, Int32x4, SignalInt32x4, Scale<int>>._(
            scale as Scale<int>,
            Layer<int, Int32x4, SignalInt32x4, Scale<int>>.fromJson(inputLayer),
            hiddenLayers.map((e) =>
                Layer<int, Int32x4, SignalInt32x4, Scale<int>>.fromJson(e)),
            Layer<int, Int32x4, SignalInt32x4, Scale<int>>.fromJson(
                outputLayer),
          );
        }
      case 'Float32x4Mod4':
        {
          return ANN<double, Float32x4, SignalFloat32x4, Scale<double>>._(
            scale as Scale<double>,
            Layer<double, Float32x4, SignalFloat32x4, Scale<double>>.fromJson(
                inputLayer),
            hiddenLayers.map((e) => Layer<double, Float32x4, SignalFloat32x4,
                Scale<double>>.fromJson(e)),
            Layer<double, Float32x4, SignalFloat32x4, Scale<double>>.fromJson(
                outputLayer),
          );
        }
      default:
        throw StateError('Unknown format: $format');
    }
  }
}

/// The configuration for the hidden layers.
class HiddenLayerConfig<N extends num, E> {
  /// Total number of neurons in the layer.
  final int neurons;

  final bool withBiasNeuron;

  /// Activation function of the layer.
  final ActivationFunction<N, E>? activationFunction;

  HiddenLayerConfig(this.neurons, this.withBiasNeuron,
      [this.activationFunction]);

  /// Returns the [activationFunction] or [def].
  ActivationFunction<A, B> getActivationFunction<A extends num, B>(
          ActivationFunction<A, B> def) =>
      (activationFunction as ActivationFunction<A, B>?) ?? def;
}

/// ANN Layer for [Int32x4] types.
///
/// (This is experimental computation layer).
class LayerInt32x4 extends Layer<int, Int32x4, SignalInt32x4, Scale<int>> {
  LayerInt32x4(int size, bool withBiasNeuron,
      [ActivationFunction<int, Int32x4> activationFunction =
          const ActivationFunctionSigmoidFastInt100()])
      : super._(SignalInt32x4((withBiasNeuron ? size + 1 : size)),
            withBiasNeuron, activationFunction);
}

/// [ANN] Layer for [Float32x4] types.
class LayerFloat32x4
    extends Layer<double, Float32x4, SignalFloat32x4, Scale<double>> {
  LayerFloat32x4(int size, bool withBiasNeuron,
      [ActivationFunction<double, Float32x4>? activationFunction])
      : super._(SignalFloat32x4((withBiasNeuron ? size + 1 : size)),
            withBiasNeuron, activationFunction ?? ActivationFunctionSigmoid());
}

/// Base class for [ANN] layers.
class Layer<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>> {
  final T _neurons;
  final bool withBiasNeuron;

  final ActivationFunction<N, E> activationFunction;

  late List<T> _weights;

  List<T> get weights => _weights;

  late List<T> _gradients;

  List<T> get gradients => _gradients;

  late List<T> _previousGradients;

  List<T> get previousGradients => _previousGradients;

  Layer<N, E, T, S>? _previousLayer;
  Layer<N, E, T, S>? _nextLayer;

  Layer<N, E, T, S>? get nextLayer => _nextLayer;

  bool get hasNextLayer => _nextLayer != null;

  Layer<N, E, T, S>? get previousLayer => _previousLayer;

  Layer._(this._neurons, this.withBiasNeuron, this.activationFunction,
      [this._weightsValues]);

  factory Layer._byType(String type, T neurons, bool withBiasNeuron,
      ActivationFunction<N, E> activationFunction,
      [List<T>? weightsValues]) {
    switch (type) {
      case 'input':
        return LayerInput<N, E, T, S>._(
            neurons, withBiasNeuron, activationFunction, weightsValues);
      case 'hidden':
        return LayerHidden<N, E, T, S>._(
            neurons, withBiasNeuron, activationFunction, weightsValues);
      case 'output':
        return LayerOutput<N, E, T, S>._(neurons, activationFunction);
      default:
        return Layer<N, E, T, S>._(
            neurons, withBiasNeuron, activationFunction, weightsValues);
    }
  }

  factory Layer.fromJson(dynamic json) {
    Map<String, dynamic> jsonMap = json is String ? parseJSON(json) : json;

    var format = jsonMap['format']! as String;
    var layerType = jsonMap['type']! as String;
    var neuronsSize = jsonMap['neurons']! as int;
    var bias = (jsonMap['bias'] ?? false) as bool;
    var activation = jsonMap['activation']! as Map<String, dynamic>;
    var weightsValues = (jsonMap['weights'] as List?)?.cast<List>();

    var activationFunction = ActivationFunction.fromJson(activation);

    switch (format) {
      case 'Float32x4':
        {
          var neurons = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
              format,
              size: neuronsSize) as SignalFloat32x4;

          var weights = weightsValues == null
              ? null
              : weightsValues
                  .map((e) =>
                      Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
                          format,
                          values: e.asDoubles()) as SignalFloat32x4)
                  .toList();

          var af = activationFunction as ActivationFunction<double, Float32x4>;

          return Layer<double, Float32x4, SignalFloat32x4,
                  Scale<double>>._byType(layerType, neurons, bias, af, weights)
              as Layer<N, E, T, S>;
        }
      case 'Int32x4':
        {
          var neurons = Signal.fromFormat<int, Int32x4, SignalInt32x4>(format,
              size: neuronsSize) as SignalInt32x4;

          var weights = weightsValues == null
              ? null
              : weightsValues
                  .map((e) => Signal.fromFormat<int, Int32x4, SignalInt32x4>(
                      format,
                      values: e.asInts()) as SignalInt32x4)
                  .toList();

          var af = activationFunction as ActivationFunction<int, Int32x4>;

          return Layer<int, Int32x4, SignalInt32x4, Scale<int>>._byType(
              layerType, neurons, bias, af, weights) as Layer<N, E, T, S>;
        }
      case 'Float32x4Mod4':
        {
          var neurons = Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
              format,
              size: neuronsSize) as SignalFloat32x4Mod4;

          var weights = weightsValues == null
              ? null
              : weightsValues
                  .map((e) =>
                      Signal.fromFormat<double, Float32x4, SignalFloat32x4>(
                          format,
                          values: e.asDoubles()) as SignalFloat32x4Mod4)
                  .toList();

          var af = activationFunction as ActivationFunction<double, Float32x4>;

          return Layer<double, Float32x4, SignalFloat32x4,
                  Scale<double>>._byType(layerType, neurons, bias, af, weights)
              as Layer<N, E, T, S>;
        }
      default:
        throw StateError('Unknown format: $format');
    }
  }

  /// The signal format of this Layer.
  String get format => neurons.format;

  /// Converts to an encoded JSON.
  String toJson({bool withIndent = false}) =>
      encodeJSON(toJsonMap(), withIndent: withIndent);

  /// Converts this Layer to a JSON [Map].
  Map<String, dynamic> toJsonMap() {
    return {
      'format': format,
      'type': layerType,
      'neurons': neurons.length,
      'bias': withBiasNeuron,
      'activation': activationFunction.toJsonMap(),
      if (this is! LayerOutput) 'weights': weights.map((e) => e.values).toList()
    };
  }

  String get layerType => '?';

  T get neurons => _neurons;

  int get length => _neurons.length;

  List<T>? _weightsValues;

  void connectTo(Layer<N, E, T, S> nextLayer,
      {Random? random, List<T>? weights}) {
    var inSize = _neurons.length;
    var outSize = nextLayer._neurons.length;

    var biasNeuronIndex = withBiasNeuron ? inSize - 1 : -1;
    var weightBiasValue =
        _neurons.toN(activationFunction.initialWeightBiasValue);

    if (weights == null && _weightsValues != null) {
      weights = _weightsValues;
      _weightsValues = null;
    }

    if (weights != null) {
      _weights = weights;
      if (_weights.length != inSize) {
        throw StateError(
            'Invalid weights length: ${_weights.length} != $inSize');
      }

      if (_weights.where((e) => e.length != outSize).isNotEmpty) {
        throw StateError(
            'Invalid weights entries length: ${_weights.map((e) => e.length).toList()} != $outSize');
      }
    } else {
      _weights = List.generate(inSize, (index) {
        T weightsToNeuron;

        if (biasNeuronIndex == index) {
          weightsToNeuron =
              neurons.createInstanceFullOfValue(outSize, weightBiasValue);
        } else {
          weightsToNeuron = neurons.createRandomInstance(outSize,
              neurons.toN(activationFunction.initialWeightScale), random);
        }

        weightsToNeuron.setExtraValues(neurons.zero);
        return weightsToNeuron;
      });
    }

    _gradients = List.generate(inSize, (index) {
      return neurons.createInstance(outSize);
    });

    _previousGradients = List.generate(inSize, (index) {
      return neurons.createInstance(outSize);
    });

    _nextLayer = nextLayer;
    nextLayer._previousLayer = this;
  }

  void resetWeights([Random? rand]) {
    if (_nextLayer == null) return;

    var initialWeightScale = activationFunction.initialWeightScale;

    for (var i = 0; i < _weights.length; ++i) {
      var weights = _weights[i];

      for (var j = 0; j < weights.entriesLength; ++j) {
        weights.setEntryWithRandomValues(
            j, weights.toN(initialWeightScale), rand);
      }
    }
  }

  void resetGradients() {
    if (_nextLayer == null) return;

    for (var i = 0; i < _gradients.length; ++i) {
      var gradients = _gradients[i];
      var prevGradients = _previousGradients[i];
      prevGradients.setAllEntriesWith(gradients);
      gradients.setAllEntriesEmpty();
    }
  }

  void setNeurons(T signal) {
    _neurons.set(signal, signal.entriesLength);
  }

  List<N> getNeurons() {
    return _neurons.values;
  }

  List<double> getNeuronsAsDouble() {
    return _neurons.valuesAsDouble;
  }

  List<String> getNeuronsAsString() {
    return _neurons.valuesAsString;
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
      LayerInput._(neurons, withBiasNeuron, activationFunction);

  LayerHidden<N, E, T, S> get asLayerHidden =>
      LayerHidden._(neurons, withBiasNeuron, activationFunction);

  LayerOutput<N, E, T, S> get asLayerOutput =>
      LayerOutput._(neurons, activationFunction);

  @override
  String toString() {
    return '$runtimeType{output: ${_neurons.valuesAsString} , withBiasNeuron: $withBiasNeuron';
  }
}

/// Layer specialized for input neurons.
class LayerInput<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  late final Layer<N, E, T, S> _nextLayerNonNull;

  LayerInput._(T neurons, bool withBiasNeuron,
      ActivationFunction<N, E> activationFunction,
      [List<T>? weightsValues])
      : super._(neurons, withBiasNeuron, activationFunction, weightsValues);

  @override
  String get layerType => 'input';

  @override
  void connectTo(Layer<N, E, T, S> nextLayer,
      {Random? random, List<T>? weights}) {
    super.connectTo(nextLayer, random: random, weights: weights);
    _nextLayerNonNull = _nextLayer!;
  }

  @override
  void activateLayer() {
    var nextLayer = _nextLayerNonNull;

    var i = _neurons.length - 1;

    if (withBiasNeuron) {
      var neuronOutput = _neurons.one;
      var weights = _weights[i];
      weights.multiplyAllValuesAddingTo(neuronOutput, nextLayer._neurons);
      --i;
    }

    for (; i >= 0; --i) {
      var neuronOutput = _neurons.getValue(i);
      var weights = _weights[i];

      weights.multiplyAllValuesAddingTo(neuronOutput, nextLayer._neurons);
    }

    nextLayer.activateLayer();
  }

  @override
  LayerInput<N, E, T, S> get asLayerInput => this;
}

/// Layer specialized for hidden neurons.
class LayerHidden<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  late final Layer<N, E, T, S> _nextLayerNonNull;

  LayerHidden._(T neurons, bool withBiasNeuron,
      ActivationFunction<N, E> activationFunction,
      [List<T>? weightsValues])
      : super._(neurons, withBiasNeuron, activationFunction, weightsValues);

  @override
  String get layerType => 'hidden';

  @override
  void connectTo(Layer<N, E, T, S> nextLayer,
      {Random? random, List<T>? weights}) {
    super.connectTo(nextLayer, random: random, weights: weights);
    _nextLayerNonNull = _nextLayer!;
  }

  @override
  void activateLayer() {
    var activationFunction = this.activationFunction.activateEntry;

    assert(_neurons.values.where((e) => e.isNaN).isEmpty);

    for (var i = _neurons.entriesLength - 1; i >= 0; --i) {
      _neurons.setEntryFilteredX4(i, activationFunction);
    }

    var nextLayer = _nextLayerNonNull;

    var i = _neurons.length - 1;

    if (withBiasNeuron) {
      var neuronOutput = _neurons.one;
      var weights = _weights[i];
      weights.multiplyAllValuesAddingTo(neuronOutput, nextLayer._neurons);
      --i;
    }

    for (; i >= 0; --i) {
      var neuronOutput = _neurons.getValue(i);
      var weights = _weights[i];

      assert(neuronOutput is! double || !neuronOutput.isNaN);
      assert(weights.values.where((e) => e.isNaN).isEmpty);

      weights.multiplyAllValuesAddingTo(neuronOutput, nextLayer._neurons);
    }

    assert(nextLayer._neurons.values.where((e) => e.isNaN).isEmpty);

    nextLayer.activateLayer();
  }

  @override
  LayerHidden<N, E, T, S> get asLayerHidden => this;
}

/// Layer specialized for output neurons.
class LayerOutput<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>> extends Layer<N, E, T, S> {
  LayerOutput._(T neurons, ActivationFunction<N, E> activationFunction)
      : super._(neurons, false, activationFunction);

  @override
  String get layerType => 'output';

  @override
  void activateLayer() {
    var activationFunction = this.activationFunction.activateEntry;

    for (var i = _neurons.entriesLength - 1; i >= 0; --i) {
      _neurons.setEntryFilteredX4(i, activationFunction);
    }
  }

  @override
  LayerOutput<N, E, T, S> get asLayerOutput => this;
}
