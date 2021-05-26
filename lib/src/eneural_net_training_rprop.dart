import 'dart:math';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

import 'eneural_net_training_parameter_strategy.dart';
import 'eneural_net_training_propagation.dart';

/// Implementation of Resilient Backpropagation (version iRProp+).
class RProp<N extends num, E, T extends Signal<N, E, T>, S extends Scale<N>,
    P extends Sample<N, E, T, S>> extends Propagation<N, E, T, S, P> {
  static const double weightMinStep = 1.0E-6;
  static const double weightMaxStep = 50.0;

  RProp(ANN<N, E, T, S> ann, SamplesSet<P> samplesSet,
      {String? subject, bool enableSelectInitialANN = false})
      : super(ann, samplesSet,
            algorithmName: 'iRProp+', subject: subject ?? samplesSet.subject) {
    this.enableSelectInitialANN = enableSelectInitialANN;
  }

  @override
  ParameterStrategy<N, E, T> createLearningRateStrategy() =>
      StaticParameterStrategy(this, 0.0);

  @override
  ParameterStrategy<N, E, T> createMomentumStrategy() =>
      StaticParameterStrategy(this, 0.0);

  @override
  double computeWeightUpdate(
    N weight,
    N weightLastUpdate,
    num gradient,
    num previousGradient,
    List<num> previousUpdateDeltas,
    List<num> noImprovementCounter,
    int weightIndex,
    N neuronOutput,
  ) {
    var previousUpdateDelta = previousUpdateDeltas[weightIndex];

    var change = (gradient * previousGradient).signWithZeroTolerance();
    var gradientSign = gradient.signWithZeroTolerance();

    // Notified by previous iteration to not change direction:
    if (previousUpdateDelta < 0) {
      previousUpdateDelta = -previousUpdateDelta;
      change = 0;
    }

    double updateDelta;
    double weightUpdate;

    if (change > 0) {
      updateDelta = previousUpdateDelta.toDouble() * 1.2;
      updateDelta = min(updateDelta, weightMaxStep);

      weightUpdate = gradientSign * updateDelta;
    } else if (change < 0) {
      updateDelta = previousUpdateDelta.toDouble() * 0.50;
      updateDelta = max(updateDelta, weightMinStep);

      // Notify to the next iteration to not change direction:
      updateDelta = -updateDelta;

      if (globalLearnError > lastGlobalLearnError) {
        weightUpdate = weightLastUpdate * -1.0;
      } else {
        weightUpdate = 0.0;
      }
    } else {
      updateDelta = previousUpdateDelta * 1.0;
      weightUpdate = gradientSign * updateDelta;
    }

    previousUpdateDeltas[weightIndex] = updateDelta;

    return weightUpdate;
  }
}
