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

/// Population / gradient-free trainers must reduce error on XOR. These are
/// stochastic and not the efficient path for tiny problems, so the bar is a
/// meaningful (>=10%) reduction rather than full convergence.
void main() {
  final scale = ScaleDouble.ZERO_TO_ONE;

  List<SampleFloat32x4> xor() => SampleFloat32x4.toListFromString(
    ['0,0=0', '0,1=1', '1,0=1', '1,1=0'],
    scale,
    true,
  );

  ANNF build({int seed = 101}) => ANN(
    scale,
    LayerFloat32x4(2, true),
    [HiddenLayerConfig(4, true)],
    LayerFloat32x4(1, false),
    random: Random(seed),
  );

  final cases = <String, TrainerF Function(ANNF, SamplesSet<SampleFloat32x4>)>{
    'EvolutionStrategy': (a, s) => EvolutionStrategy(a, s, random: Random(1)),
    'SeparableCMAES': (a, s) => SeparableCMAES(a, s, random: Random(1)),
    'GeneticAlgorithm': (a, s) => GeneticAlgorithm(a, s, random: Random(1)),
    'ParticleSwarm': (a, s) => ParticleSwarm(a, s, random: Random(1)),
    'DifferentialEvolution': (a, s) =>
        DifferentialEvolution(a, s, random: Random(1)),
    'SimulatedAnnealing': (a, s) => SimulatedAnnealing(a, s, random: Random(1)),
  };

  group('Population trainers reduce XOR error', () {
    cases.forEach((name, factory) {
      test(name, () {
        final t = factory(build(), SamplesSet(xor(), subject: 'xor'))
          ..logEnabled = false;
        final before = t.ann.computeSamplesGlobalError(xor());
        t.train(300, 0.0);
        final after = t.ann.computeSamplesGlobalError(xor());
        expect(after, lessThan(before * 0.9), reason: '$name should improve');
      });
    });
  });
}
