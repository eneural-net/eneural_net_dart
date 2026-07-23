import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

/// A `Float32x4` ANN, the most common configuration of the library.
typedef ANNFloat32x4 = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// A `Float32x4` layer.
typedef LayerF32x4 = Layer<double, Float32x4, SignalFloat32x4, Scale<double>>;

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;

  ANNFloat32x4 buildANN({
    int inputs = 2,
    List<int> hidden = const [3],
    int outputs = 1,
    ActivationFunction<double, Float32x4>? activationFunction,
    Random? random,
  }) {
    var af = activationFunction ?? ActivationFunctionSigmoid();

    return ANN(
      scaleDouble,
      LayerFloat32x4(inputs, true, af),
      hidden.map((n) => HiddenLayerConfig(n, true)).toList(),
      LayerFloat32x4(outputs, false, af),
      random: random,
    );
  }

  group('ANN: structure', () {
    test('layers and sizes', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      expect(ann.inputSize, equals(3), reason: '2 inputs + bias');
      expect(ann.outputSize, equals(1));
      expect(ann.allLayers.length, equals(3));
      expect(ann.hiddenLayers.length, equals(1));
      expect(ann.allLayersNeuronsSize, equals([3, 4, 1]));
      expect(ann.format, equals('Float32x4'));
      expect(ann.scale, equals(scaleDouble));
    });

    test('without bias neurons', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, false), [
        HiddenLayerConfig(3, false),
      ], LayerFloat32x4(1, false));

      expect(ann.allLayersNeuronsSize, equals([2, 3, 1]));
      expect(ann.inputLayer.withBiasNeuron, isFalse);
    });

    test('multiple hidden layers', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(3, true), [
        HiddenLayerConfig(5, true),
        HiddenLayerConfig(4, true),
      ], LayerFloat32x4(2, false));

      expect(ann.hiddenLayers.length, equals(2));
      expect(ann.allLayersNeuronsSize, equals([4, 6, 5, 2]));
      expect(ann.allLayers.length, equals(4));
    });

    test('layer types', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      expect(ann.inputLayer, isA<LayerInput>());
      expect(ann.hiddenLayers.first, isA<LayerHidden>());
      expect(ann.outputLayer, isA<LayerOutput>());

      expect(ann.inputLayer.layerType, equals('input'));
      expect(ann.hiddenLayers.first.layerType, equals('hidden'));
      expect(ann.outputLayer.layerType, equals('output'));
    });

    test('layers are chained', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      expect(ann.inputLayer.hasNextLayer, isTrue);
      expect(ann.inputLayer.nextLayer, same(ann.hiddenLayers.first));
      expect(ann.hiddenLayers.first.nextLayer, same(ann.outputLayer));
      expect(ann.outputLayer.hasNextLayer, isFalse);
      expect(ann.outputLayer.nextLayer, isNull);

      expect(ann.hiddenLayers.first.previousLayer, same(ann.inputLayer));
      expect(ann.outputLayer.previousLayer, same(ann.hiddenLayers.first));
      expect(ann.inputLayer.previousLayer, isNull);
    });

    test('rejects a bias neuron on the output layer', () {
      expect(
        () => ANN(scaleDouble, LayerFloat32x4(2, true), [
          HiddenLayerConfig(3, true),
        ], LayerFloat32x4(1, true)),
        throwsA(isA<StateError>()),
      );
    });

    test('toString describes the topology', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      var str = ann.toString();
      expect(str, contains('2+'));
      expect(str, contains('3+'));
      expect(str, contains('ScaleDouble'));

      expect(ann.inputLayer.toString(), contains('withBiasNeuron: true'));
    });
  });

  group('ANN: weights', () {
    test('allWeightsLength matches allWeights', () {
      var ann = buildANN();

      expect(ann.allWeights.length, equals(ann.allWeightsLength));

      // input(3 neurons x 4 next) + hidden(4 neurons x 1 next)
      expect(ann.allWeightsLength, equals(3 * 4 + 4 * 1));
    });

    test('allWeights can be read back after being set', () {
      var ann = buildANN();

      var weights = List<double>.generate(
        ann.allWeightsLength,
        (i) => (i + 1) * 0.1,
      );

      ann.allWeights = weights;

      var readBack = ann.allWeights;
      expect(readBack.length, equals(weights.length));

      for (var i = 0; i < weights.length; ++i) {
        expect(readBack[i], closeTo(weights[i], 1e-6), reason: 'weight $i');
      }
    });

    test('allWeightsHashcode changes with the weights', () {
      var ann = buildANN();
      var hash1 = ann.allWeightsHashcode;

      ann.allWeights = List<double>.filled(ann.allWeightsLength, 0.5);
      var hash2 = ann.allWeightsHashcode;

      expect(hash1, isNot(equals(hash2)));
      expect(ann.allWeightsHashcode, equals(hash2), reason: 'stable');
    });

    test('resetWeights changes the weights', () {
      var ann = buildANN();
      var before = ann.allWeights;

      ann.resetWeights(Random(1));
      var after = ann.allWeights;

      expect(after.length, equals(before.length));
      expect(after, isNot(equals(before)));
    });

    test('resetWeights is reproducible with the same seed', () {
      var ann1 = buildANN();
      var ann2 = buildANN();

      ann1.resetWeights(Random(7));
      ann2.resetWeights(Random(7));

      expect(ann1.allWeights, equals(ann2.allWeights));
    });

    test('a seeded ANN is reproducible', () {
      var a = buildANN(random: Random(11));
      var b = buildANN(random: Random(11));

      expect(a.allWeights, equals(b.allWeights));
    });

    test('the weights beyond the layer size stay zero', () {
      // 5 output neurons -> 8 values of capacity, 3 of them unused.
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(5, true),
      ], LayerFloat32x4(1, false));

      void checkPadding(String when) {
        for (var weights in ann.inputLayer.weights) {
          for (var i = weights.length; i < weights.capacity; ++i) {
            expect(
              weights.getValue(i),
              equals(0.0),
              reason: 'extra weight $i must be zero $when',
            );
          }
        }
      }

      checkPadding('after building');

      ann.resetWeights(Random(3));
      checkPadding('after resetWeights');

      ann.allWeights = List<double>.filled(ann.allWeightsLength, 0.5);
      checkPadding('after setting allWeights');
    });

    test('resetGradients moves gradients to previousGradients', () {
      var ann = buildANN();
      var layer = ann.inputLayer;

      layer.gradients.first.setAllEntriesWithValue(2.0);

      ann.resetGradients();

      expect(layer.gradients.first.values.every((v) => v == 0), isTrue);
      expect(layer.previousGradients.first.values.every((v) => v == 2), isTrue);
    });
  });

  group('ANN: activation', () {
    test('produces an output of the expected size', () {
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(2, false));

      ann.activate(SignalFloat32x4.from([0.5, 0.5]));

      expect(ann.output.length, equals(2));
      expect(ann.outputAsDouble.length, equals(2));
      expect(ann.output.every((v) => !v.isNaN), isTrue);
    });

    test('a sigmoid output stays in 0..1', () {
      var ann = buildANN();

      for (var input in [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
      ]) {
        ann.activate(SignalFloat32x4.from(input));
        expect(
          ann.output.every((v) => v >= 0 && v <= 1),
          isTrue,
          reason: 'input $input -> ${ann.output}',
        );
      }
    });

    test('is deterministic', () {
      var ann = buildANN(random: Random(5));
      var input = SignalFloat32x4.from([0.3, 0.7]);

      ann.activate(input);
      var out1 = List<double>.from(ann.outputAsDouble);

      ann.activate(input);
      var out2 = List<double>.from(ann.outputAsDouble);

      expect(out1, equals(out2));
    });

    test('different inputs give different outputs', () {
      var ann = buildANN(random: Random(5));

      ann.activate(SignalFloat32x4.from([0.0, 0.0]));
      var out1 = List<double>.from(ann.outputAsDouble);

      ann.activate(SignalFloat32x4.from([1.0, 1.0]));
      var out2 = List<double>.from(ann.outputAsDouble);

      expect(out1, isNot(equals(out2)));
    });

    test('outputDenormalized applies the scale', () {
      var scale = ScaleDouble(0, 100);
      var ann = ANN(scale, LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      ann.activate(SignalFloat32x4.from([0.5, 0.5]));

      var normalized = ann.output.first;
      var denormalized = ann.outputDenormalized.first;

      expect(denormalized, closeTo(normalized * 100, 1e-6));
    });

    test('computeSamplesActivations returns one output per sample', () {
      var ann = buildANN();
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
        scaleDouble,
        true,
      );

      var activations = ann.computeSamplesActivations(samples);

      expect(activations.length, equals(4));
      expect(activations.every((a) => a.length == 1), isTrue);
    });

    test('computeSamplesErrors and the global error', () {
      var ann = buildANN();
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
        scaleDouble,
        true,
      );

      var errors = ann.computeSamplesErrors(samples);

      expect(errors.length, equals(4));
      expect(errors.every((e) => e >= 0), isTrue);

      var globalError = ann.computeSamplesGlobalError(samples);
      var mean = errors.reduce((a, b) => a + b) / errors.length;

      expect(globalError, closeTo(mean, 1e-12));
    });
  });

  group('ANN: layers', () {
    test('getNeurons variants', () {
      var ann = buildANN();
      ann.activate(SignalFloat32x4.from([0.5, 0.5]));

      var layer = ann.outputLayer;

      expect(layer.getNeurons().length, equals(1));
      expect(layer.getNeuronsAsDouble().length, equals(1));
      expect(layer.getNeuronsAsString().length, equals(1));
      expect(
        layer.getNeuronsDenormalized(ScaleDouble(0, 10)).first,
        closeTo(layer.getNeurons().first * 10, 1e-6),
      );
    });

    test('resetLayer/resetNetwork zero the neurons', () {
      var ann = buildANN();
      ann.activate(SignalFloat32x4.from([1.0, 1.0]));

      ann.outputLayer.resetLayer();
      expect(ann.outputLayer.getNeurons().every((v) => v == 0), isTrue);

      ann.activate(SignalFloat32x4.from([1.0, 1.0]));
      ann.inputLayer.resetNetwork();
      expect(ann.outputLayer.getNeurons().every((v) => v == 0), isTrue);
    });

    test('the base Layer cannot be activated', () {
      var layer = LayerF32x4.fromJson({
        'format': 'Float32x4',
        'type': 'unknown',
        'neurons': 2,
        'bias': false,
        'activation': ActivationFunctionSigmoid().toJsonMap(),
        'weights': <List<double>>[],
      });

      expect(() => layer.activateLayer(), throwsA(isA<UnsupportedError>()));
    });

    test('asLayerInput/asLayerHidden/asLayerOutput', () {
      var ann = buildANN();

      expect(ann.inputLayer.asLayerInput, same(ann.inputLayer));
      expect(
        ann.hiddenLayers.first.asLayerHidden,
        same(ann.hiddenLayers.first),
      );
      expect(ann.outputLayer.asLayerOutput, same(ann.outputLayer));

      // Converting a plain layer creates a new specialized one:
      var plain = LayerFloat32x4(2, true);
      expect(plain.asLayerInput, isA<LayerInput>());
      expect(plain.asLayerHidden, isA<LayerHidden>());

      // An output layer can't have a bias neuron:
      expect(() => plain.asLayerOutput, throwsA(isA<StateError>()));
      expect(LayerFloat32x4(2, false).asLayerOutput, isA<LayerOutput>());
    });

    test('connectTo validates the given weights', () {
      var a = LayerFloat32x4(2, false);
      var b = LayerFloat32x4(3, false);

      expect(
        () => a.connectTo(
          b,
          weights: [
            SignalFloat32x4.from([1, 2, 3]),
          ],
        ),
        throwsA(isA<StateError>()),
        reason: 'wrong number of weight signals',
      );

      var c = LayerFloat32x4(2, false);
      var d = LayerFloat32x4(3, false);
      expect(
        () => c.connectTo(
          d,
          weights: [
            SignalFloat32x4.from([1, 2]),
            SignalFloat32x4.from([1, 2]),
          ],
        ),
        throwsA(isA<StateError>()),
        reason: 'wrong weight signal length',
      );
    });

    test('connectTo accepts valid weights', () {
      var a = LayerFloat32x4(2, false);
      var b = LayerFloat32x4(3, false);

      a.connectTo(
        b,
        weights: [
          SignalFloat32x4.from([1, 2, 3]),
          SignalFloat32x4.from([4, 5, 6]),
        ],
      );

      expect(a.weights.length, equals(2));
      expect(a.weights.first.values, equals([1, 2, 3]));
      expect(a.hasNextLayer, isTrue);
    });

    test('the bias neuron gets the activation function bias weight', () {
      var af = ActivationFunctionSigmoid();
      var ann = ANN(scaleDouble, LayerFloat32x4(2, true, af), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false, af));

      // Index of the bias neuron of the input layer:
      var biasWeights = ann.inputLayer.weights[ann.inputLayer.length - 1];

      expect(
        biasWeights.getValues(ann.hiddenLayers.first.length),
        everyElement(equals(af.initialWeightBiasValue)),
      );
    });

    test('HiddenLayerConfig', () {
      var config = HiddenLayerConfig(5, true);

      expect(config.neurons, equals(5));
      expect(config.withBiasNeuron, isTrue);
      expect(config.activationFunction, isNull);

      var def = ActivationFunctionSigmoid();
      expect(config.getActivationFunction(def), same(def));

      var custom = ActivationFunctionSigmoidFast();
      var config2 = HiddenLayerConfig(5, true, custom);
      expect(config2.getActivationFunction(def), same(custom));
    });

    test('a hidden layer can have its own activation function', () {
      var ann = ANN(
        scaleDouble,
        LayerFloat32x4(2, true, ActivationFunctionSigmoid()),
        [HiddenLayerConfig(3, true, ActivationFunctionSigmoidFast())],
        LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
      );

      expect(
        ann.hiddenLayers.first.activationFunction,
        isA<ActivationFunctionSigmoidFast>(),
      );
      expect(
        ann.outputLayer.activationFunction,
        isA<ActivationFunctionSigmoid>(),
      );
    });

    test('an input layer with a Linear function is the hidden default', () {
      var ann = ANN(
        scaleDouble,
        LayerFloat32x4(2, true, ActivationFunctionLinear()),
        [HiddenLayerConfig(3, true)],
        LayerFloat32x4(1, false, ActivationFunctionSigmoid()),
      );

      // `Linear` is the only one with the `input` scope, so it becomes the
      // default for the hidden layers.
      expect(
        ann.hiddenLayers.first.activationFunction,
        isA<ActivationFunctionLinear>(),
      );
    });
  });

  group('ANN: JSON', () {
    test('round-trips a Float32x4 ANN', () {
      var ann = buildANN(random: Random(3));
      var json = ann.toJson();

      var decoded = ANN.fromJson(json);

      expect(decoded.allLayersNeuronsSize, equals(ann.allLayersNeuronsSize));
      expect(decoded.allWeights, equals(ann.allWeights));
      expect(decoded.format, equals(ann.format));
      expect(decoded.scale, equals(ann.scale));
      expect(decoded.toJson(), equals(json));
    });

    test('round-trips an Int32x4 ANN', () {
      var ann = ANN(ScaleInt.ZERO_TO_ONE, LayerInt32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerInt32x4(1, false));

      var json = ann.toJson();
      var decoded = ANN.fromJson(json);

      expect(decoded.allLayersNeuronsSize, equals(ann.allLayersNeuronsSize));
      expect(decoded.allWeights, equals(ann.allWeights));
      expect(decoded.toJson(), equals(json));
    });

    test('round-trips a zoomable scale', () {
      var ann = ANN(ScaleZoomableDouble(0, 100, 10), LayerFloat32x4(2, true), [
        HiddenLayerConfig(3, true),
      ], LayerFloat32x4(1, false));

      var decoded = ANN.fromJson(ann.toJson());

      expect(decoded.scale, isA<ScaleZoomableDouble>());
      expect(decoded.scale, equals(ann.scale));
    });

    test('round-trips the activation functions', () {
      var ann = ANN(
        scaleDouble,
        LayerFloat32x4(2, true, ActivationFunctionSigmoidBoundedFast(scale: 4)),
        [HiddenLayerConfig(3, true)],
        LayerFloat32x4(
          1,
          false,
          ActivationFunctionSigmoidBoundedFast(scale: 4),
        ),
      );

      var decoded = ANN.fromJson(ann.toJson());
      var af =
          decoded.outputLayer.activationFunction
              as ActivationFunctionSigmoidBoundedFast;

      expect(af.scale, equals(4.0));
    });

    test('a restored ANN computes the same outputs', () {
      var ann = buildANN(random: Random(9));
      var samples = SampleFloat32x4.toListFromString(
        ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
        scaleDouble,
        true,
      );

      var expected = ann.computeSamplesActivations(samples);

      var decoded = ANN.fromJson(ann.toJson());
      var actual = decoded.computeSamplesActivations(samples);

      expect(actual, equals(expected));
    });

    test('multiple hidden layers round-trip', () {
      var ann = ANN(
        scaleDouble,
        LayerFloat32x4(3, true),
        [HiddenLayerConfig(5, true), HiddenLayerConfig(4, true)],
        LayerFloat32x4(2, false),
        random: Random(4),
      );

      var decoded = ANN.fromJson(ann.toJson());

      expect(decoded.allLayersNeuronsSize, equals(ann.allLayersNeuronsSize));
      expect(decoded.allWeights, equals(ann.allWeights));
      expect(decoded.hiddenLayers.length, equals(2));
    });

    test('toJson accepts an indent flag', () {
      var ann = buildANN();

      expect(ann.toJson(withIndent: true), contains('\n'));
      expect(ann.toJson(withIndent: false), isNot(contains('\n')));
    });

    test('toJsonMap has the expected shape', () {
      var ann = buildANN();
      var map = ann.toJsonMap();

      expect(map['format'], equals('Float32x4'));
      expect(map['scale'], isA<Map>());
      expect(map['layers'], isA<List>());
      expect((map['layers'] as List).length, equals(3));
    });

    test('an unknown format throws', () {
      var ann = buildANN();
      var map = ann.toJsonMap();
      map['format'] = 'Nope';

      expect(() => ANN.fromJson(map), throwsA(isA<StateError>()));
    });

    test('an output layer has no weights in JSON', () {
      var ann = buildANN();
      var layers = (ann.toJsonMap()['layers'] as List).cast<Map>();

      expect(layers[0].containsKey('weights'), isTrue);
      expect(layers[1].containsKey('weights'), isTrue);
      expect(layers[2].containsKey('weights'), isFalse);
    });

    test('an unconnected layer can be serialized', () {
      var layer = LayerFloat32x4(2, true);

      // No `connectTo` was called, so there are no weights yet:
      expect(() => layer.toJsonMap(), returnsNormally);

      var map = layer.toJsonMap();
      expect(map['neurons'], equals(3));
      expect(map['bias'], isTrue);
      expect(map.containsKey('weights'), isFalse);

      expect(() => layer.toJson(), returnsNormally);
    });

    test('Layer.fromJson round-trips', () {
      var ann = buildANN(random: Random(2));
      var layerJson = ann.inputLayer.toJsonMap();

      var decoded = LayerF32x4.fromJson(layerJson);

      expect(decoded, isA<LayerInput>());
      expect(decoded.length, equals(ann.inputLayer.length));
      expect(decoded.withBiasNeuron, equals(ann.inputLayer.withBiasNeuron));
      expect(decoded.format, equals('Float32x4'));
    });

    test('Layer.fromJson decodes from a String', () {
      var ann = buildANN();
      var decoded = LayerF32x4.fromJson(ann.inputLayer.toJson());
      expect(decoded, isA<LayerInput>());
    });

    test('Layer.fromJson rejects an unknown format', () {
      expect(
        () => LayerF32x4.fromJson({
          'format': 'Nope',
          'type': 'input',
          'neurons': 2,
          'bias': false,
          'activation': ActivationFunctionSigmoid().toJsonMap(),
        }),
        throwsA(isA<StateError>()),
      );
    });

    test('Layer.fromJson for the Float32x4Mod4 format', () {
      var decoded = LayerF32x4.fromJson({
        'format': 'Float32x4Mod4',
        'type': 'output',
        'neurons': 2,
        'bias': false,
        'activation': ActivationFunctionSigmoid().toJsonMap(),
      });

      expect(decoded, isA<LayerOutput>());
      expect(decoded.format, equals('Float32x4Mod4'));
    });
  });
}
