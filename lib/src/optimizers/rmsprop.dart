import 'package:eneural_net/eneural_net.dart';

/// RMSProp optimizer.
///
///   v = ρ·v + (1−ρ)·g²
///   Δw = lr · g / (√v + ε)
class RMSProp<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final double rho;
  final double epsilon;

  late final List<List<T>> _v;

  RMSProp(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.01,
    this.rho = 0.9,
    this.epsilon = 1e-8,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'RMSProp',
         baseLearningRate: learningRate,
         subject: subject,
       ) {
    _v = createWeightStateBuffers();
  }

  @override
  E updateWeightEntry({
    required int layerIndex,
    required int neuronIndex,
    required int entryIndex,
    required E weight,
    required E gradient,
    required E previousGradient,
    required E neuronOutput,
  }) {
    final si = signalInstance;
    final vSig = _v[layerIndex][neuronIndex];

    final g2 = si.entryOperationMultiply(gradient, gradient);
    final vNew = si.entryOperationSum(
      si.entryOperationScale(vSig.getEntry(entryIndex), rho),
      si.entryOperationScale(g2, 1 - rho),
    );
    vSig.setEntry(entryIndex, vNew);

    final denom = si.entryOperationScalarAdd(
      si.entryOperationSqrt(vNew),
      epsilon,
    );
    return si.entryOperationScale(
      si.entryOperationDivide(gradient, denom),
      learningRate,
    );
  }
}
