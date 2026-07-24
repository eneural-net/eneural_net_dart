import 'package:eneural_net/eneural_net.dart';

/// AdaDelta optimizer (Zeiler, 2012). Needs no global learning rate.
///
///   Eg² = ρ·Eg² + (1−ρ)·g²
///   Δw  = (√(EΔ²+ε) / √(Eg²+ε)) · g
///   EΔ² = ρ·EΔ² + (1−ρ)·Δw²
class AdaDelta<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final double rho;
  final double epsilon;

  late final List<List<T>> _eg2;
  late final List<List<T>> _edx2;

  AdaDelta(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    this.rho = 0.95,
    this.epsilon = 1e-6,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'AdaDelta',
         baseLearningRate: 1.0,
         subject: subject,
       ) {
    _eg2 = createWeightStateBuffers();
    _edx2 = createWeightStateBuffers();
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
    final eg2Sig = _eg2[layerIndex][neuronIndex];
    final edx2Sig = _edx2[layerIndex][neuronIndex];

    final g2 = si.entryOperationMultiply(gradient, gradient);
    final eg2 = si.entryOperationSum(
      si.entryOperationScale(eg2Sig.getEntry(entryIndex), rho),
      si.entryOperationScale(g2, 1 - rho),
    );
    eg2Sig.setEntry(entryIndex, eg2);

    // rms(EΔ²) / rms(Eg²) · g
    final rmsDx = si.entryOperationSqrt(
      si.entryOperationScalarAdd(edx2Sig.getEntry(entryIndex), epsilon),
    );
    final rmsG = si.entryOperationSqrt(
      si.entryOperationScalarAdd(eg2, epsilon),
    );
    final delta = si.entryOperationMultiply(
      si.entryOperationDivide(rmsDx, rmsG),
      gradient,
    );

    // EΔ² = ρ·EΔ² + (1−ρ)·Δw²
    final dx2 = si.entryOperationMultiply(delta, delta);
    final edx2 = si.entryOperationSum(
      si.entryOperationScale(edx2Sig.getEntry(entryIndex), rho),
      si.entryOperationScale(dx2, 1 - rho),
    );
    edx2Sig.setEntry(entryIndex, edx2);

    return delta;
  }
}
