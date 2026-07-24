import 'dart:math';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

/// Resilient Backpropagation variants (Riedmiller & Braun; Igel & Hüsken).
///
/// Sign-based per-weight step-size adaptation:
///  - gradient sign unchanged → step ×η⁺ (capped at [stepMax])
///  - sign flipped            → step ×η⁻ (floored at [stepMin]); the flipped
///    step is neutralized next iteration (encoded as a negative stored step)
///  - sign zero               → plain step
///
/// Four backtracking behaviors on a sign flip:
///  - [RPropVariant.rpropPlus]  : revert the last weight update
///  - [RPropVariant.rpropMinus] : no backtracking (keep stepping)
///  - [RPropVariant.iRpropPlus] : revert only if the total error increased
///  - [RPropVariant.iRpropMinus]: no weight step on the flip
///
/// Note: [RProp] already provides iRProp+ via the classic seam; this class
/// offers the full family on the [GradientOptimizer] seam.
enum RPropVariant { rpropPlus, rpropMinus, iRpropPlus, iRpropMinus }

class ResilientPropagation<
  N extends num,
  E,
  T extends Signal<N, E, T>,
  S extends Scale<N>,
  P extends Sample<N, E, T, S>
>
    extends GradientOptimizer<N, E, T, S, P> {
  final RPropVariant variant;
  final double stepInit;
  final double stepMin;
  final double stepMax;
  final double etaPlus;
  final double etaMinus;

  late final List<List<T>> _step; // signed: negative = "neutralize next step"
  late final List<List<T>> _lastUpdate;

  ResilientPropagation(
    ANN<N, E, T, S> ann,
    SamplesSet<P> samplesSet, {
    this.variant = RPropVariant.iRpropPlus,
    this.stepInit = 0.10,
    this.stepMin = 1e-6,
    this.stepMax = 50.0,
    this.etaPlus = 1.2,
    this.etaMinus = 0.5,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         algorithmName: 'RProp:${variant.name}',
         baseLearningRate: 0.0,
         subject: subject,
       ) {
    _step = createWeightStateBuffers(fill: stepInit);
    _lastUpdate = createWeightStateBuffers();
  }

  static const double _tol = 1e-20;

  /// Returns `[weightUpdate, newStep]` for one lane.
  List<double> _lane(double g, double gPrev, double step, double lastUpdate) {
    var change = (g * gPrev).signWithZeroTolerance(_tol);
    final gSign = g.signWithZeroTolerance(_tol);

    if (step < 0) {
      step = -step;
      change = 0;
    }

    double newStep;
    double update;

    if (change > 0) {
      newStep = min(step * etaPlus, stepMax);
      update = gSign * newStep;
    } else if (change < 0) {
      newStep = max(step * etaMinus, stepMin);
      switch (variant) {
        case RPropVariant.rpropPlus:
          update = -lastUpdate;
          newStep = -newStep;
        case RPropVariant.iRpropPlus:
          update = globalLearnError > lastGlobalLearnError ? -lastUpdate : 0.0;
          newStep = -newStep;
        case RPropVariant.iRpropMinus:
          update = 0.0;
          newStep = -newStep;
        case RPropVariant.rpropMinus:
          update = gSign * newStep; // keep stepping, no neutralization flag
      }
    } else {
      newStep = step;
      update = gSign * newStep;
    }

    return [update, newStep];
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
    final stepSig = _step[layerIndex][neuronIndex];
    final lastSig = _lastUpdate[layerIndex][neuronIndex];
    final stepEntry = stepSig.getEntry(entryIndex);
    final lastEntry = lastSig.getEntry(entryIndex);

    final updates = <double>[];
    final steps = <double>[];
    for (var lane = 0; lane < 4; ++lane) {
      final g = si.getValueFromEntry(gradient, lane).toDouble();
      final gp = si.getValueFromEntry(previousGradient, lane).toDouble();
      final st = si.getValueFromEntry(stepEntry, lane).toDouble();
      final lu = si.getValueFromEntry(lastEntry, lane).toDouble();
      final r = _lane(g, gp, st, lu);
      updates.add(r[0]);
      steps.add(r[1]);
    }

    final updateEntry = si.createEntry4(
      si.toN(updates[0]),
      si.toN(updates[1]),
      si.toN(updates[2]),
      si.toN(updates[3]),
    );
    stepSig.setEntry(
      entryIndex,
      si.createEntry4(
        si.toN(steps[0]),
        si.toN(steps[1]),
        si.toN(steps[2]),
        si.toN(steps[3]),
      ),
    );
    lastSig.setEntry(entryIndex, updateEntry);
    return updateEntry;
  }
}
