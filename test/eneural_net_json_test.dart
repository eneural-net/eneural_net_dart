import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  var scaleDouble = ScaleDouble.ZERO_TO_ONE;
  var samplesXOR_Float32x4 = SampleFloat32x4.toListFromString([
    '0,0=0',
    '0,1=1',
    '1,0=1',
    '1,1=0',
  ], scaleDouble, true);

  group('JSON', () {
    setUp(() {
      print('================================================================');
    });

    test('Backpropagation + ActivationFunctionSigmoid', () {
      var activationFunction = ActivationFunctionSigmoid();

      var samplesSet = SamplesSet(samplesXOR_Float32x4, subject: 'xor');

      var ann1 = ANN(
          scaleDouble,
          LayerFloat32x4(2, true, activationFunction),
          [HiddenLayerConfig(3, true)],
          LayerFloat32x4(1, false, activationFunction));

      var ann1Json1 = ann1.toJson(withIndent: true);
      print(ann1Json1);

      var ann2 = ANN.fromJson(ann1Json1);

      expect(ann1.allLayersNeuronsSize, equals(ann2.allLayersNeuronsSize));
      expect(ann1.allWeights, equals(ann2.allWeights));

      expect(ann2.toJson(), equals(ann1Json1));

      var training = Backpropagation(ann1, samplesSet);

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
  });
}
