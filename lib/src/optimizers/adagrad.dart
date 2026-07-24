import 'package:eneural_net/eneural_net.dart';

/// AdaGrad optimizer.
///
///   G = G + g²
///   Δw = lr · g / (√G + ε)
class AdaGrad<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final double epsilon;

  late final List<List<T>> _g2;

  AdaGrad(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.05,
    this.epsilon = 1e-8,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'AdaGrad',
         baseLearningRate: learningRate,
         subject: subject,
       ) {
    _g2 = createWeightStateBuffers();
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
    final gSig = _g2[layerIndex][neuronIndex];

    final accum = si.entryOperationSum(
      gSig.getEntry(entryIndex),
      si.entryOperationMultiply(gradient, gradient),
    );
    gSig.setEntry(entryIndex, accum);

    final denom = si.entryOperationScalarAdd(
      si.entryOperationSqrt(accum),
      epsilon,
    );
    return si.entryOperationScale(
      si.entryOperationDivide(gradient, denom),
      learningRate,
    );
  }
}
