import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

/// Concrete trainer type used by the name-based registry (Float32x4 networks).
typedef TrainingD =
    Training<
      double,
      Float32x4,
      SignalFloat32x4,
      Scale<double>,
      SampleFloat32x4
    >;

/// ANN type used by the registry.
typedef AnnD = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;
typedef SamplesD = SamplesSet<SampleFloat32x4>;

/// Builds a trainer from a parameter map.
typedef TrainingBuilderFn =
    TrainingD Function(AnnD ann, SamplesD samples, Map<String, dynamic> params);

double _d(Map p, String k, double def) => (p[k] as num?)?.toDouble() ?? def;
int _i(Map p, String k, int def) => (p[k] as num?)?.toInt() ?? def;

final Map<String, TrainingBuilderFn> _registry = {
  'backpropagation': (a, s, p) => Backpropagation(a, s),
  'rprop': (a, s, p) => RProp(a, s),
  'sgd': (a, s, p) => SGD(
    a,
    s,
    learningRate: _d(p, 'learningRate', 0.1),
    momentum: _d(p, 'momentum', 0.0),
    batchSize: _i(p, 'batchSize', 0),
  ),
  'adam': (a, s, p) => Adam(
    a,
    s,
    learningRate: _d(p, 'learningRate', 0.001),
    weightDecay: _d(p, 'weightDecay', 0.0),
    batchSize: _i(p, 'batchSize', 0),
  ),
  'adamw': (a, s, p) => Adam(
    a,
    s,
    learningRate: _d(p, 'learningRate', 0.001),
    weightDecay: _d(p, 'weightDecay', 0.01),
  ),
  'nadam': (a, s, p) =>
      Adam(a, s, learningRate: _d(p, 'learningRate', 0.001), nesterov: true),
  'amsgrad': (a, s, p) =>
      Adam(a, s, learningRate: _d(p, 'learningRate', 0.001), amsgrad: true),
  'rmsprop': (a, s, p) =>
      RMSProp(a, s, learningRate: _d(p, 'learningRate', 0.01)),
  'adagrad': (a, s, p) =>
      AdaGrad(a, s, learningRate: _d(p, 'learningRate', 0.05)),
  'adadelta': (a, s, p) => AdaDelta(a, s),
  'quickprop': (a, s, p) =>
      Quickprop(a, s, learningRate: _d(p, 'learningRate', 0.5)),
  'lion': (a, s, p) => Lion(a, s, learningRate: _d(p, 'learningRate', 0.001)),
  'levenbergmarquardt': (a, s, p) => LevenbergMarquardt(a, s),
  'conjugategradient': (a, s, p) => ConjugateGradient(a, s),
  'lbfgs': (a, s, p) => LBFGS(a, s),
  'evolutionstrategy': (a, s, p) => EvolutionStrategy(a, s),
  'geneticalgorithm': (a, s, p) => GeneticAlgorithm(a, s),
  'particleswarm': (a, s, p) => ParticleSwarm(a, s),
  'differentialevolution': (a, s, p) => DifferentialEvolution(a, s),
  'simulatedannealing': (a, s, p) => SimulatedAnnealing(a, s),
};

/// Registers a trainer builder under [name] (case-insensitive).
void registerTraining(String name, TrainingBuilderFn builder) {
  _registry[name.toLowerCase()] = builder;
}

/// The registered algorithm names.
Iterable<String> registeredTrainings() => _registry.keys;

/// Builds a trainer by [name] with optional [params]. Throws if unknown.
TrainingD trainingByName(
  String name,
  AnnD ann,
  SamplesD samples, {
  Map<String, dynamic> params = const {},
}) {
  final builder = _registry[name.toLowerCase()];
  if (builder == null) {
    throw StateError('Unknown training algorithm: "$name"');
  }
  return builder(ann, samples, params);
}

/// Serializes a resumable checkpoint of [training]: the ANN weights plus, for
/// gradient optimizers, the optimizer state buffers and step counter.
Map<String, dynamic> saveTrainingCheckpoint(TrainingD training) {
  final map = <String, dynamic>{
    'algorithm': training.algorithmName,
    'annWeights': training.ann.allWeights,
    'globalError': training.globalError,
  };
  if (training is GradientOptimizer) {
    final opt = training as GradientOptimizer;
    map['optimizerStep'] = opt.optimizerStep;
    map['optimizerState'] = opt.saveOptimizerState();
    // Also capture the accumulated gradients (become `previousGradient` next
    // epoch) and the epoch error-tracking, so optimizers that read those
    // (Quickprop, iRProp+) resume exactly.
    map['gradients'] = opt.saveGradients();
    map['globalLearnError'] = opt.globalLearnError;
    map['lastGlobalLearnError'] = opt.lastGlobalLearnError;
  }
  return map;
}

/// Restores a checkpoint produced by [saveTrainingCheckpoint] into [training]
/// (which must have the same topology/algorithm/hyperparameters).
void restoreTrainingCheckpoint(
  TrainingD training,
  Map<String, dynamic> checkpoint,
) {
  final w = (checkpoint['annWeights'] as List)
      .map((e) => (e as num).toDouble())
      .toList();
  training.ann.allWeights = w;
  if (training is GradientOptimizer && checkpoint['optimizerState'] != null) {
    final opt = training as GradientOptimizer;
    opt.optimizerStep = (checkpoint['optimizerStep'] as num).toInt();
    opt.loadOptimizerState(checkpoint['optimizerState'] as List);
    if (checkpoint['gradients'] != null) {
      opt.loadGradients(checkpoint['gradients'] as List);
    }
    if (checkpoint['globalLearnError'] != null) {
      opt.restoreGlobalLearnErrors(
        (checkpoint['lastGlobalLearnError'] as num).toDouble(),
        (checkpoint['globalLearnError'] as num).toDouble(),
      );
    }
  }
}
