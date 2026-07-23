import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;
  var samplesXorFloat32x4 = SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scaleDouble,
    true,
  );

  ANN<double, Float32x4, SignalFloat32x4, Scale<double>> buildANN({
    ActivationFunction<double, Float32x4>? activationFunction,
    int seed = 101,
  }) {
    var af = activationFunction ?? ActivationFunctionSigmoid();

    return ANN(
      scaleDouble,
      LayerFloat32x4(2, true, af),
      [HiddenLayerConfig(3, true)],
      LayerFloat32x4(1, false, af),
      random: Random(seed),
    );
  }

  group('JSON', () {
    setUp(() {
      print('================================================================');
    });

    test('Backpropagation + ActivationFunctionSigmoid', () {
      var samplesSet = SamplesSet(samplesXorFloat32x4, subject: 'xor');

      var ann1 = buildANN();

      var ann1Json1 = ann1.toJson(withIndent: true);
      print(ann1Json1);

      var ann2 = ANN.fromJson(ann1Json1);

      expect(ann1.allLayersNeuronsSize, equals(ann2.allLayersNeuronsSize));
      expect(ann1.allWeights, equals(ann2.allWeights));

      expect(ann2.toJson(), equals(ann1Json1));

      var training = Backpropagation(ann1, samplesSet);
      training.logEnabled = false;

      var trainError = training.train(1000, 0.01);

      print('trainError: $trainError');
      expect(trainError < 0.40, isTrue);

      var annGlobalError1 = ann1.computeSamplesGlobalError(samplesSet.samples);
      print('annGlobalError1: $annGlobalError1');

      var ann1Json2 = ann1.toJson(withIndent: true);

      var ann3 = ANN.fromJson(ann1Json2);
      expect(ann3.toJson(), equals(ann1Json2));

      var annGlobalError3 = ann3.computeSamplesGlobalError(samplesSet.samples);
      print('annGlobalError3: $annGlobalError3');

      expect(annGlobalError3, equals(annGlobalError1));
    });

    test('round-trips every Float32x4 activation function', () {
      var functions = <ActivationFunction<double, Float32x4>>[
        ActivationFunctionSigmoid(),
        ActivationFunctionSigmoidFast(),
        ActivationFunctionSigmoidBoundedFast(),
        ActivationFunctionSigmoidBoundedFast(scale: 3),
        ActivationFunctionLinear(),
      ];

      for (var af in functions) {
        var ann = buildANN(activationFunction: af);
        var json = ann.toJson();

        var decoded = ANN.fromJson(json);

        expect(decoded.toJson(), equals(json), reason: af.name);
        expect(decoded.allWeights, equals(ann.allWeights), reason: af.name);
        expect(
          decoded.outputLayer.activationFunction.runtimeType,
          equals(af.runtimeType),
          reason: af.name,
        );
      }
    });

    test('the JSON is stable across repeated round-trips', () {
      var ann = buildANN();

      var json1 = ann.toJson();
      var json2 = ANN.fromJson(json1).toJson();
      var json3 = ANN.fromJson(json2).toJson();

      expect(json2, equals(json1));
      expect(json3, equals(json1));
    });

    test('a trained ANN keeps its predictions after a round-trip', () {
      var ann = buildANN();
      var samplesSet = SamplesSet(samplesXorFloat32x4, subject: 'xor');

      var training = Backpropagation(ann, samplesSet);
      training.logEnabled = false;
      training.train(500, 0.001);

      var expected = ann.computeSamplesActivations(samplesXorFloat32x4);

      var decoded =
          ANN.fromJson(ann.toJson())
              as ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

      expect(
        decoded.computeSamplesActivations(samplesXorFloat32x4),
        equals(expected),
      );
    });
  });
}
