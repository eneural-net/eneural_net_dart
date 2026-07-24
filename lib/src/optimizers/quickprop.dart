import 'package:eneural_net/eneural_net.dart';

/// Quickprop optimizer (Fahlman, 1988) — a second-order-ish method that fits a
/// parabola through the current and previous gradient of each weight.
///
/// Using the library's gradient convention (`g` = error-reducing direction) and
/// `Δw(t-1)` = the previous weight update:
///   Δw(t) = (g / (g_prev − g)) · Δw(t-1)          (bounded by [maxGrowthFactor])
/// with a bootstrap/gradient term `lr·g` on the first step or while descending.
///
/// Sign/branch-heavy, so implemented per lane (not vectorized).
class Quickprop<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  /// Maximum growth factor μ (a step never exceeds μ× the previous step).
  final double maxGrowthFactor;

  late final List<List<T>> _prevDelta;

  Quickprop(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.5,
    this.maxGrowthFactor = 1.75,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'Quickprop',
         baseLearningRate: learningRate,
         subject: subject,
       ) {
    _prevDelta = createWeightStateBuffers();
  }

  static const double _tiny = 1e-10;

  double _lane(double g, double gPrev, double prevDelta, double lr) {
    if (prevDelta.abs() < _tiny) {
      return lr * g;
    }
    final denom = gPrev - g;
    double delta;
    if (denom.abs() < _tiny) {
      delta = maxGrowthFactor * prevDelta;
    } else {
      delta = (g / denom) * prevDelta;
      final cap = maxGrowthFactor * prevDelta.abs();
      if (delta > cap) {
        delta = cap;
      } else if (delta < -cap) {
        delta = -cap;
      }
    }
    // Add a gradient-descent term while still moving downhill.
    if (g * prevDelta > 0) delta += lr * g;
    return delta;
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
    final lr = learningRate;
    final pdSig = _prevDelta[layerIndex][neuronIndex];
    final pdEntry = pdSig.getEntry(entryIndex);

    final d = <double>[];
    for (var lane = 0; lane < 4; ++lane) {
      final g = si.getValueFromEntry(gradient, lane).toDouble();
      final gp = si.getValueFromEntry(previousGradient, lane).toDouble();
      final pd = si.getValueFromEntry(pdEntry, lane).toDouble();
      d.add(_lane(g, gp, pd, lr));
    }

    final deltaEntry = si.createEntry4(
      si.toN(d[0]),
      si.toN(d[1]),
      si.toN(d[2]),
      si.toN(d[3]),
    );
    pdSig.setEntry(entryIndex, deltaEntry);
    return deltaEntry;
  }
}
