import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:eneural_net/eneural_net_extensions.dart';

/// Sample type accepted by the population/non-gradient trainers (Float32x4).
typedef PopulationSample =
    Sample<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Base for gradient-free trainers that optimize the flat weight vector
/// ([ANN.allWeights]) using [ANN.computeSamplesGlobalError] as the fitness.
/// Each [learn] call performs one generation/iteration and installs the best
/// genome found into the [ANN].
abstract class PopulationTrainer<P extends PopulationSample>
    extends Training<double, Float32x4, SignalFloat32x4, Scale<double>, P> {
  final Random random;
  late final int dim;

  double _bestFitness = double.infinity;
  List<double>? _bestGenome;

  PopulationTrainer(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet,
    String algorithmName, {
    Random? random,
    String? subject,
  }) : random = random ?? Random(),
       super(ann, samplesSet, algorithmName, subject: subject) {
    dim = ann.allWeightsLength;
  }

  @override
  String get parameters => 'dim: $dim';

  /// Sets the weights and returns the fitness (mean squared error).
  double evaluate(List<double> genome) {
    ann.allWeights = genome;
    final f = ann.computeSamplesGlobalError(samples);
    if (f < _bestFitness) {
      _bestFitness = f;
      _bestGenome = List<double>.of(genome);
    }
    return f;
  }

  List<double> gaussianGenome(List<double> center, double sigma) =>
      List.generate(dim, (i) => center[i] + sigma * random.nextGaussian());

  /// Installs the best genome found so far and returns whether the target error
  /// has been reached.
  bool finishGeneration(double targetGlobalError) {
    if (_bestGenome != null) ann.allWeights = _bestGenome!;
    return _bestFitness <= targetGlobalError;
  }
}

/// (μ, λ) Evolution Strategy with a global step size (1/5-success adaptation).
class EvolutionStrategy<P extends PopulationSample>
    extends PopulationTrainer<P> {
  final int mu;
  final int lambda;
  double sigma;

  List<double>? _mean;

  EvolutionStrategy(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.mu = 10,
    this.lambda = 40,
    this.sigma = 0.5,
    Random? random,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'EvolutionStrategy',
         random: random,
         subject: subject,
       );

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final mean = _mean ??= List<double>.of(ann.allWeights);

    final offspring = List.generate(lambda, (_) => gaussianGenome(mean, sigma))
      ..sort((a, b) => evaluate(a).compareTo(evaluate(b)));

    final bestF = evaluate(offspring.first);

    // Recombine the μ best into the new mean.
    final newMean = List<double>.filled(dim, 0);
    for (var k = 0; k < mu; ++k) {
      final g = offspring[k];
      for (var i = 0; i < dim; ++i) {
        newMean[i] += g[i] / mu;
      }
    }
    final meanF = evaluate(newMean);
    _mean = meanF < bestF ? newMean : offspring.first;

    // 1/5-ish step-size adaptation.
    sigma *= meanF < bestF ? 1.1 : 0.92;
    sigma = sigma.clamp(1e-4, 10.0);

    return finishGeneration(targetGlobalError);
  }
}

/// Separable (diagonal) CMA-ES: per-coordinate step sizes updated from the
/// variance of the selected steps.
class SeparableCMAES<P extends PopulationSample> extends PopulationTrainer<P> {
  final int lambda;
  final int mu;
  double sigma;

  List<double>? _mean;
  late final List<double> _std;

  SeparableCMAES(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    int? lambda,
    int? mu,
    this.sigma = 0.5,
    Random? random,
    String? subject,
  }) : lambda = lambda ?? (4 + (3 * log(ann.allWeightsLength + 1)).floor()),
       mu =
           mu ??
           ((lambda ?? (4 + (3 * log(ann.allWeightsLength + 1)).floor())) ~/ 2),
       super(
         ann,
         samplesSet,
         'SeparableCMAES',
         random: random,
         subject: subject,
       ) {
    _std = List<double>.filled(dim, 1.0);
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final mean = _mean ??= List<double>.of(ann.allWeights);

    // Sample and score.
    final steps = List.generate(
      lambda,
      (_) => List.generate(dim, (i) => _std[i] * random.nextGaussian()),
    );
    final pop = steps
        .map((z) => List.generate(dim, (i) => mean[i] + sigma * z[i]))
        .toList();
    final idx = List<int>.generate(lambda, (i) => i)
      ..sort((a, b) => evaluate(pop[a]).compareTo(evaluate(pop[b])));

    // Recombine μ best.
    final newMean = List<double>.filled(dim, 0);
    for (var k = 0; k < mu; ++k) {
      final g = pop[idx[k]];
      for (var i = 0; i < dim; ++i) {
        newMean[i] += g[i] / mu;
      }
    }

    // Update per-coordinate std from the selected steps' second moment.
    for (var i = 0; i < dim; ++i) {
      var v = 0.0;
      for (var k = 0; k < mu; ++k) {
        final z = steps[idx[k]][i];
        v += z * z / mu;
      }
      _std[i] = (0.7 * _std[i] + 0.3 * sqrt(v + 1e-12)).clamp(1e-4, 100.0);
    }
    _mean = newMean;
    sigma = (sigma * 0.98).clamp(1e-4, 10.0);

    return finishGeneration(targetGlobalError);
  }
}

/// Genetic Algorithm: tournament selection, blend crossover, Gaussian mutation,
/// elitism.
class GeneticAlgorithm<P extends PopulationSample>
    extends PopulationTrainer<P> {
  final int populationSize;
  final double mutationRate;
  final double mutationSigma;
  final int elitism;

  List<List<double>>? _pop;

  GeneticAlgorithm(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.populationSize = 50,
    this.mutationRate = 0.1,
    this.mutationSigma = 0.3,
    this.elitism = 2,
    Random? random,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'GeneticAlgorithm',
         random: random,
         subject: subject,
       );

  List<double> _tournament(List<List<double>> pop, List<double> fit) {
    var best = random.nextInt(pop.length);
    for (var i = 0; i < 2; ++i) {
      final c = random.nextInt(pop.length);
      if (fit[c] < fit[best]) best = c;
    }
    return pop[best];
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final base = ann.allWeights;
    final pop = _pop ??= [
      List<double>.of(base),
      for (var i = 1; i < populationSize; ++i) gaussianGenome(base, 0.5),
    ];

    final fit = pop.map(evaluate).toList();
    final order = List<int>.generate(pop.length, (i) => i)
      ..sort((a, b) => fit[a].compareTo(fit[b]));

    final next = <List<double>>[];
    for (var e = 0; e < elitism; ++e) {
      next.add(List<double>.of(pop[order[e]]));
    }
    while (next.length < populationSize) {
      final p1 = _tournament(pop, fit);
      final p2 = _tournament(pop, fit);
      final child = List<double>.generate(dim, (i) {
        final a = random.nextDouble();
        var v = a * p1[i] + (1 - a) * p2[i];
        if (random.nextDouble() < mutationRate) {
          v += mutationSigma * random.nextGaussian();
        }
        return v;
      });
      next.add(child);
    }
    _pop = next;

    return finishGeneration(targetGlobalError);
  }
}

/// Particle Swarm Optimization.
class ParticleSwarm<P extends PopulationSample> extends PopulationTrainer<P> {
  final int swarmSize;
  final double inertia;
  final double cognitive;
  final double social;

  List<List<double>>? _pos;
  late List<List<double>> _vel;
  late List<List<double>> _pbest;
  late List<double> _pbestFit;
  List<double>? _gbest;
  double _gbestFit = double.infinity;

  ParticleSwarm(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.swarmSize = 30,
    this.inertia = 0.7,
    this.cognitive = 1.5,
    this.social = 1.5,
    Random? random,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'ParticleSwarm',
         random: random,
         subject: subject,
       );

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final base = ann.allWeights;
    if (_pos == null) {
      _pos = [
        List<double>.of(base),
        for (var i = 1; i < swarmSize; ++i) gaussianGenome(base, 0.5),
      ];
      _vel = List.generate(swarmSize, (_) => List<double>.filled(dim, 0));
      _pbest = _pos!.map((p) => List<double>.of(p)).toList();
      _pbestFit = _pos!.map(evaluate).toList();
      for (var i = 0; i < swarmSize; ++i) {
        if (_pbestFit[i] < _gbestFit) {
          _gbestFit = _pbestFit[i];
          _gbest = List<double>.of(_pos![i]);
        }
      }
    }

    final pos = _pos!;
    final gbest = _gbest!;
    for (var p = 0; p < swarmSize; ++p) {
      for (var i = 0; i < dim; ++i) {
        final r1 = random.nextDouble();
        final r2 = random.nextDouble();
        _vel[p][i] =
            inertia * _vel[p][i] +
            cognitive * r1 * (_pbest[p][i] - pos[p][i]) +
            social * r2 * (gbest[i] - pos[p][i]);
        pos[p][i] += _vel[p][i];
      }
      final f = evaluate(pos[p]);
      if (f < _pbestFit[p]) {
        _pbestFit[p] = f;
        _pbest[p] = List<double>.of(pos[p]);
        if (f < _gbestFit) {
          _gbestFit = f;
          _gbest = List<double>.of(pos[p]);
        }
      }
    }

    return finishGeneration(targetGlobalError);
  }
}

/// Differential Evolution (DE/rand/1/bin).
class DifferentialEvolution<P extends PopulationSample>
    extends PopulationTrainer<P> {
  final int populationSize;
  final double differentialWeight; // F
  final double crossoverRate; // CR

  List<List<double>>? _pop;
  late List<double> _fit;

  DifferentialEvolution(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.populationSize = 40,
    this.differentialWeight = 0.8,
    this.crossoverRate = 0.9,
    Random? random,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'DifferentialEvolution',
         random: random,
         subject: subject,
       );

  int _other(int not) {
    var r = random.nextInt(populationSize);
    while (r == not) {
      r = random.nextInt(populationSize);
    }
    return r;
  }

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final base = ann.allWeights;
    if (_pop == null) {
      _pop = [
        List<double>.of(base),
        for (var i = 1; i < populationSize; ++i) gaussianGenome(base, 0.5),
      ];
      _fit = _pop!.map(evaluate).toList();
    }

    final pop = _pop!;
    for (var i = 0; i < populationSize; ++i) {
      final a = _other(i);
      final b = _other(i);
      final c = _other(i);
      final jRand = random.nextInt(dim);
      final trial = List<double>.generate(dim, (j) {
        if (random.nextDouble() < crossoverRate || j == jRand) {
          return pop[a][j] + differentialWeight * (pop[b][j] - pop[c][j]);
        }
        return pop[i][j];
      });
      final f = evaluate(trial);
      if (f <= _fit[i]) {
        pop[i] = trial;
        _fit[i] = f;
      }
    }

    return finishGeneration(targetGlobalError);
  }
}

/// Simulated Annealing (single candidate, geometric cooling).
class SimulatedAnnealing<P extends PopulationSample>
    extends PopulationTrainer<P> {
  double temperature;
  final double coolingRate;
  final double stepSize;

  List<double>? _current;
  double _currentFit = double.infinity;

  SimulatedAnnealing(
    ANN<double, Float32x4, SignalFloat32x4, Scale<double>> ann,
    SamplesSet<P> samplesSet, {
    this.temperature = 1.0,
    this.coolingRate = 0.98,
    this.stepSize = 0.3,
    Random? random,
    String? subject,
  }) : super(
         ann,
         samplesSet,
         'SimulatedAnnealing',
         random: random,
         subject: subject,
       );

  @override
  bool learn(List<P> samples, double targetGlobalError) {
    final current = _current ??= List<double>.of(ann.allWeights);
    if (_currentFit.isInfinite) _currentFit = evaluate(current);

    final neighbor = gaussianGenome(current, stepSize * (temperature + 0.05));
    final f = evaluate(neighbor);
    final delta = f - _currentFit;

    if (delta < 0 || random.nextDouble() < exp(-delta / (temperature + 1e-9))) {
      _current = neighbor;
      _currentFit = f;
    }
    temperature = (temperature * coolingRate).clamp(1e-4, double.infinity);

    return finishGeneration(targetGlobalError);
  }
}
