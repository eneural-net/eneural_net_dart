import 'package:eneural_net/eneural_net.dart';

/// Lion optimizer (EvoLved Sign Momentum, Chen et al., 2023).
///
/// Sign-based and memory-light (one momentum buffer). Add-convention:
///   c  = β1·m + (1−β1)·g
///   Δw = lr · sign(c)              (− lr·wd·w  when weightDecay > 0)
///   m  = β2·m + (1−β2)·g
class Lion<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final double beta1;
  final double beta2;

  late final List<List<T>> _m;

  Lion(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.99,
    double weightDecay = 0.0,
    int batchSize = 0,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'Lion',
         baseLearningRate: learningRate,
         weightDecay: weightDecay,
         batchSize: batchSize,
         subject: subject,
       ) {
    _m = createWeightStateBuffers();
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
    final mSig = _m[layerIndex][neuronIndex];
    final mOld = mSig.getEntry(entryIndex);

    // c = β1·m + (1−β1)·g ; step direction = sign(c)
    final c = si.entryOperationSum(
      si.entryOperationScale(mOld, beta1),
      si.entryOperationScale(gradient, 1 - beta1),
    );
    // Decoupled weight decay is applied uniformly by the base GradientOptimizer.
    final delta = si.entryOperationScale(
      si.entryOperationSign(c),
      learningRate,
    );

    // m = β2·m + (1−β2)·g
    final mNew = si.entryOperationSum(
      si.entryOperationScale(mOld, beta2),
      si.entryOperationScale(gradient, 1 - beta2),
    );
    mSig.setEntry(entryIndex, mNew);

    return delta;
  }
}
