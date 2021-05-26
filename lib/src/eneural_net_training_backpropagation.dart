import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/src/eneural_net_training_propagation.dart';

/// Implementation of Backpropagation training algorithm.
class Backpropagation<
    N extends num,
    E,
    T extends Signal<N, E, T>,
    S extends Scale<N>,
    P extends Sample<N, E, T, S>> extends Propagation<N, E, T, S, P> {
  Backpropagation(ANN<N, E, T, S> ann, SamplesSet<P> samplesSet,
      {String? subject})
      : super(ann, samplesSet,
            algorithmName: 'Backpropagation',
            subject: subject ?? samplesSet.subject);

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

    var deltaMomentum = previousUpdateDelta * momentum;
    var gradientLearn = learningRate * gradient;

    var delta = gradientLearn + deltaMomentum;

    previousUpdateDeltas[weightIndex] = delta;

    return delta;
  }

  @override
  E computeEntryWeightUpdateSIMD(
    E weight,
    E weightLastUpdate,
    E gradient,
    E previousGradient,
    T previousUpdateDeltas,
    T noImprovementCounter,
    int weightsEntryIndex,
    E neuronOutput,
  ) {
    var previousUpdateDelta = previousUpdateDeltas.getEntry(weightsEntryIndex);

    var deltaMomentum = signalInstance.entryOperationMultiply(
        previousUpdateDelta, momentumEntry);
    var gradientLearn =
        signalInstance.entryOperationMultiply(learningRateEntry, gradient);

    var delta = signalInstance.entryOperationSum(gradientLearn, deltaMomentum);

    previousUpdateDeltas.setEntry(weightsEntryIndex, delta);

    return delta;
  }

  @override
  E computeEntryWeightUpdate(
    E weight,
    E weightLastUpdate,
    E gradient,
    E previousGradient,
    T previousUpdateDeltas,
    T noImprovementCounter,
    int weightsEntryIndex,
    E neuronOutput,
  ) =>
      computeEntryWeightUpdateSIMD(
          weight,
          weightLastUpdate,
          gradient,
          previousGradient,
          previousUpdateDeltas,
          noImprovementCounter,
          weightsEntryIndex,
          neuronOutput);
}
