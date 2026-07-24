import 'dart:math' as math;

import 'package:eneural_net/eneural_net.dart';

/// Adam optimizer (Kingma & Ba, 2014), with optional AdamW decoupled weight
/// decay, AMSGrad, and Nadam (Nesterov) variants.
///
/// Per weight, using the library's gradient-sign convention (the accumulated
/// gradient already points in the error-reducing direction, so the update is
/// ADDED to the weight):
///
///   m = β1·m + (1−β1)·g
///   v = β2·v + (1−β2)·g²
///   m̂ = m / (1−β1ᵗ) ;  v̂ = v / (1−β2ᵗ)
///   Δw = lr · m̂ / (√v̂ + ε)          (− lr·wd·w  when decoupledWeightDecay)
class Adam<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final double beta1;
  final double beta2;
  final double epsilon;

  /// Use the AMSGrad variant (max-of-past second moments).
  final bool amsgrad;

  /// Use the Nadam (Nesterov-accelerated) variant.
  final bool nesterov;

  late final List<List<T>> _m;
  late final List<List<T>> _v;
  late final List<List<T>>? _vMax;

  Adam(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    double learningRate = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
    double weightDecay = 0.0,
    this.amsgrad = false,
    this.nesterov = false,
    int batchSize = 0,
    double gradientClip = 0.0,
    LearningRateScheduleBuilder<N, E, T>? lrSchedule,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: nesterov
             ? 'Nadam'
             : (weightDecay > 0 ? 'AdamW' : 'Adam'),
         baseLearningRate: learningRate,
         batchSize: batchSize,
         weightDecay: weightDecay,
         gradientClip: gradientClip,
         lrSchedule: lrSchedule,
         subject: subject,
       ) {
    _m = createWeightStateBuffers();
    _v = createWeightStateBuffers();
    _vMax = amsgrad ? createWeightStateBuffers() : null;
  }

  int _cachedStep = -1;
  double _biasCorr1 = 1.0;
  double _biasCorr2 = 1.0;

  void _updateBiasCorrections() {
    final t = optimizerStep;
    if (t == _cachedStep) return;
    _cachedStep = t;
    _biasCorr1 = 1.0 - math.pow(beta1, t).toDouble();
    _biasCorr2 = 1.0 - math.pow(beta2, t).toDouble();
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
    _updateBiasCorrections();
    final si = signalInstance;

    final mSig = _m[layerIndex][neuronIndex];
    final vSig = _v[layerIndex][neuronIndex];

    final mOld = mSig.getEntry(entryIndex);
    final vOld = vSig.getEntry(entryIndex);

    // m = β1·m + (1−β1)·g
    final mNew = si.entryOperationSum(
      si.entryOperationScale(mOld, beta1),
      si.entryOperationScale(gradient, 1 - beta1),
    );
    // v = β2·v + (1−β2)·g²
    final g2 = si.entryOperationMultiply(gradient, gradient);
    final vNew = si.entryOperationSum(
      si.entryOperationScale(vOld, beta2),
      si.entryOperationScale(g2, 1 - beta2),
    );

    mSig.setEntry(entryIndex, mNew);
    vSig.setEntry(entryIndex, vNew);

    // Second-moment used for the denominator (AMSGrad keeps the running max).
    E vForDenom = vNew;
    if (_vMax != null) {
      final vMaxSig = _vMax[layerIndex][neuronIndex];
      final vMaxNew = si.entryOperationMax(vMaxSig.getEntry(entryIndex), vNew);
      vMaxSig.setEntry(entryIndex, vMaxNew);
      vForDenom = vMaxNew;
    }

    // Bias-corrected estimates.
    final mHat = si.entryOperationScale(mNew, 1 / _biasCorr1);
    final vHat = si.entryOperationScale(vForDenom, 1 / _biasCorr2);

    // Numerator (Nadam blends in the current gradient).
    E numerator = mHat;
    if (nesterov) {
      final gHat = si.entryOperationScale(gradient, (1 - beta1) / _biasCorr1);
      numerator = si.entryOperationSum(
        si.entryOperationScale(mHat, beta1),
        gHat,
      );
    }

    // Δw = lr · numerator / (√v̂ + ε).  Decoupled weight decay (AdamW) is
    // applied uniformly by the base GradientOptimizer.
    final denom = si.entryOperationScalarAdd(
      si.entryOperationSqrt(vHat),
      epsilon,
    );
    return si.entryOperationScale(
      si.entryOperationDivide(numerator, denom),
      learningRate,
    );
  }
}
