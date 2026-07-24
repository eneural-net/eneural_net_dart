import 'package:eneural_net/eneural_net.dart';

/// Stochastic Gradient Descent with optional (classic or Nesterov) momentum.
///
/// Add-convention (the accumulated gradient `g` already points in the
/// error-reducing direction):
///   v = μ·v + g
///   Δw = lr·v                    (classic momentum)
///   Δw = lr·(μ·v + g)            (Nesterov)
///
/// With `momentum = 0` this is plain SGD (`Δw = lr·g`). Pair with a small
/// `batchSize` for mini-batch/online SGD.
class SGD<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  /// Momentum coefficient (named `momentumFactor` to avoid clashing with
  /// [Propagation.momentum], which is the momentum-strategy value).
  final double momentumFactor;
  final bool nesterov;

  late final List<List<T>>? _velocity;

  SGD(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.1,
    double momentum = 0.0,
    this.nesterov = false,
    int batchSize = 0,
    LearningRateScheduleBuilder<N, E, T>? lrSchedule,
    String? subject,
  }) : momentumFactor = momentum,
       super(
         ann,
         samplesSet,
         algorithmName: momentum > 0
             ? (nesterov ? 'NesterovSGD' : 'SGD+M')
             : 'SGD',
         baseLearningRate: learningRate,
         batchSize: batchSize,
         lrSchedule: lrSchedule,
         subject: subject,
       ) {
    _velocity = momentum > 0 ? createWeightStateBuffers() : null;
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
    if (_velocity == null) {
      return si.entryOperationScale(gradient, learningRate);
    }

    final vSig = _velocity[layerIndex][neuronIndex];
    final vNew = si.entryOperationSum(
      si.entryOperationScale(vSig.getEntry(entryIndex), momentumFactor),
      gradient,
    );
    vSig.setEntry(entryIndex, vNew);

    final step = nesterov
        ? si.entryOperationSum(
            si.entryOperationScale(vNew, momentumFactor),
            gradient,
          )
        : vNew;
    return si.entryOperationScale(step, learningRate);
  }
}
