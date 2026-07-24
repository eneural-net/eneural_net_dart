import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;
typedef TrainerF =
    Training<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

/// Convergence tests for the M1 gradient optimizers. Each must learn XOR.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({int hidden = 4, int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(hidden, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  final cases = <String, TrainerF Function(ANNF, SamplesSet<SampleFloat32x4>)>{
    'SGD': (a, s) => SGD(a, s, learningRate: 0.5),
    'SGD+Momentum': (a, s) => SGD(a, s, learningRate: 0.3, momentum: 0.9),
    'Nesterov': (a, s) =>
        SGD(a, s, learningRate: 0.3, momentum: 0.9, nesterov: true),
    'Adam': (a, s) => Adam(a, s, learningRate: 0.05),
    'AdamW': (a, s) => Adam(a, s, learningRate: 0.05, weightDecay: 0.001),
    'AMSGrad': (a, s) => Adam(a, s, learningRate: 0.05, amsgrad: true),
    'Nadam': (a, s) => Adam(a, s, learningRate: 0.05, nesterov: true),
    'RMSProp': (a, s) => RMSProp(a, s, learningRate: 0.02),
    'AdaGrad': (a, s) => AdaGrad(a, s, learningRate: 0.1),
    'AdaDelta': (a, s) => AdaDelta(a, s),
    'Lion': (a, s) => Lion(a, s, learningRate: 0.02),
    'Quickprop': (a, s) => Quickprop(a, s, learningRate: 0.3),
    'RProp+': (a, s) =>
        ResilientPropagation(a, s, variant: RPropVariant.rpropPlus),
    'RProp-': (a, s) =>
        ResilientPropagation(a, s, variant: RPropVariant.rpropMinus),
    'iRProp+': (a, s) =>
        ResilientPropagation(a, s, variant: RPropVariant.iRpropPlus),
    'iRProp-': (a, s) =>
        ResilientPropagation(a, s, variant: RPropVariant.iRpropMinus),
  };

  group('Optimizers converge on XOR', () {
    cases.forEach((name, factory) {
      test(name, () {
        final trainer = factory(build(), SamplesSet(xor(), subject: 'xor'))
          ..logEnabled = false
          ..enableSelectInitialANN = false; // deterministic initial weights

        final ok = trainer.trainUntilGlobalError(
          targetGlobalError: 1e-3,
          maxEpochs: 20000,
        );

        expect(ok, isTrue, reason: '$name should converge');
        expect(trainer.globalError, lessThan(1e-3));

        for (final s in xor()) {
          trainer.ann.activate(s.input);
          final out = trainer.ann.outputAsDouble.first;
          final expected = s.output.valuesAsDouble.first;
          expect((out - expected).abs(), lessThan(0.1), reason: '$name $name');
        }
      });
    });
  });
}
