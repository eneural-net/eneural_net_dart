import 'dart:math';

import 'package:eneural_net/eneural_net.dart';

import 'eneural_net_training_propagation.dart';

/// Base class for training parameter strategy.
abstract class ParameterStrategy<N extends num, E, T extends Signal<N, E, T>> {
  final Propagation<N, E, T, dynamic, dynamic> _propagation;

  ParameterStrategy(this._propagation);

  void initializeValue();

  double get initialValue;

  void resetValue() {
    setValue(initialValue);
  }

  double get value;

  E get valueEntry;

  void setValue(double value);

  void updateValue();

  E createValueEntry(double value) {
    return _propagation.signalInstance.createEntryFullOf(
      _propagation.signalInstance.toN(value),
    );
  }
}

/// A parameter strategy with a static/constant value.
class StaticParameterStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends ParameterStrategy<N, E, T> {
  double _value;

  StaticParameterStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation, [
    this._value = 0.0,
  ]) : super(propagation) {
    _valueEntry = createValueEntry(_value);
  }

  @override
  double get value => _value;

  @override
  E get valueEntry => _valueEntry;

  late E _valueEntry;

  @override
  void setValue(double value) {
    if (_value != value) {
      _value = value;
      _valueEntry = _propagation.signalInstance.createEntryFullOf(
        _propagation.signalInstance.toN(value),
      );
    }
  }

  @override
  double get initialValue => _value;

  late double _initialValue;

  @override
  void initializeValue() {
    _initialValue = _value;
  }

  @override
  void resetValue() {
    setValue(_initialValue);
  }

  @override
  void updateValue() {}
}

/// A parameter strategy with a value proportional to the current global error.
class ProportionalToErrorStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends ParameterStrategy<N, E, T> {
  final double minValue;

  final double maxValue;

  final double zero;
  final double multiplier;

  ProportionalToErrorStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation, {
    this.minValue = 0.0,
    this.maxValue = 1.0,
    this.zero = 0.0,
    this.multiplier = 1.0,
  }) : super(propagation);

  double _value = -1;

  @override
  double get value => _value;

  @override
  E get valueEntry => _learningRateEntry;

  late E _learningRateEntry;

  @override
  void setValue(double value) {
    if (_value != value) {
      _value = value;
      _learningRateEntry = createValueEntry(value);
    }
  }

  double _initialValue = 1.0;

  @override
  double get initialValue => _initialValue;

  @override
  void initializeValue() {
    _initialValue = computeValue(1.0);
    setValue(_initialValue);
  }

  @override
  void resetValue() {
    setValue(_initialValue);
  }

  @override
  void updateValue() {
    var value = computeValue(_propagation.globalLearnError);
    setValue(value);
  }

  double computeValue(double error) {
    return (zero + (error * multiplier)).clamp(minValue, maxValue);
  }
}

/// Specialized strategy for learning rate.
class LearningRateStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends ParameterStrategy<N, E, T> {
  final double multiplier;

  LearningRateStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation, {
    this.multiplier = 1.0,
  }) : super(propagation);

  double _learningRate = -1;

  @override
  double get value => _learningRate;

  @override
  E get valueEntry => _learningRateEntry;

  late E _learningRateEntry;

  @override
  void setValue(double value) {
    if (_learningRate != value) {
      _learningRate = value;
      _learningRateEntry = createValueEntry(value);
    }
  }

  double _initialValue = 0;

  @override
  double get initialValue => _initialValue;

  @override
  void initializeValue() {
    var trainingSamplesSize = _propagation.trainingSamplesSize;
    // Before the training is initialized `trainingSamplesSize` is 0.
    // Avoids an `Infinity` learning rate:
    _initialValue = trainingSamplesSize > 0
        ? (1 / trainingSamplesSize) * multiplier
        : multiplier;
    setValue(_initialValue);
    _noLearnCount = 0;
    _noLearnNearZeroCount = 0;
  }

  @override
  void resetValue() {
    setValue(_initialValue);
    _noLearnCount = 0;
    _noLearnNearZeroCount = 0;
  }

  int _noLearnCount = 0;
  int _noLearnNearZeroCount = 0;

  @override
  void updateValue() {
    var lastImprovement =
        _propagation.globalLearnError - _propagation.lastGlobalLearnError;

    if (lastImprovement > 0) {
      // The error grew: decrease the learning rate every 10 epochs.
      _noLearnNearZeroCount = 0;

      if (++_noLearnCount % 10 == 0) {
        var learningRate = _learningRate * 0.90;
        learningRate = max(learningRate, _initialValue / 1000);
        setValue(learningRate);
      }
    } else {
      _noLearnCount = 0;

      var lastImprovementRatio =
          lastImprovement / _propagation.lastGlobalLearnError;

      if (lastImprovementRatio > -1.0E-4) {
        // The error is barely improving: recover the learning rate
        // every 10 epochs.
        if (++_noLearnNearZeroCount % 10 == 0) {
          var learningRate = _learningRate * 1.10;

          if (learningRate > _initialValue) {
            learningRate = _initialValue;
          }

          setValue(learningRate);
          _noLearnNearZeroCount = 0;
        }
      } else {
        _noLearnNearZeroCount = 0;
      }
    }
  }
}

/// Specialized strategy for momentum.
class MomentumRateStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends ParameterStrategy<N, E, T> {
  MomentumRateStrategy(Propagation<N, E, T, dynamic, dynamic> propagation)
    : super(propagation);

  @override
  void initializeValue() {
    setValue(0.0);
    _lastMomentum = 0;
  }

  @override
  double get initialValue => 0.0;

  double _momentum = -1;

  @override
  double get value => _momentum;

  late E _momentumEntry;

  @override
  E get valueEntry => _momentumEntry;

  @override
  void setValue(double value) {
    if (_momentum != value) {
      _momentum = value;
      _momentumEntry = createValueEntry(value);
    }
  }

  @override
  void resetValue() {
    setValue(0.0);
    _lastMomentum = 0;
  }

  int _lastMomentum = 0;

  @override
  void updateValue() {
    var lastImprovement =
        _propagation.globalLearnError - _propagation.lastGlobalLearnError;

    if (lastImprovement <= 0) {
      var lastImprovementRatio =
          lastImprovement / _propagation.lastGlobalLearnError;

      if (lastImprovementRatio > -1.0E-4) {
        setValue(0.0);
        return;
      }
    }

    ++_lastMomentum;

    if (_lastMomentum > 10) {
      _lastMomentum = 0;

      var momentum = _momentum;

      if (momentum == 0.0) {
        momentum = 0.10;
      }

      momentum *= 1.01;
      if (momentum > 1) momentum = 0.101;

      setValue(momentum);
    }
  }
}

/// Base for epoch-based learning-rate schedules. The value is recomputed each
/// epoch from [Training.trainedEpochs] via [computeValue].
abstract class LearningRateScheduleStrategy<
  N extends num,
  E,
  T extends Signal<N, E, T>
>
    extends ParameterStrategy<N, E, T> {
  final double baseValue;
  double _value;
  late E _valueEntry;

  /// Epochs elapsed (incremented on each [updateValue]); independent of the
  /// training block bookkeeping.
  int _epoch = 0;

  LearningRateScheduleStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation,
    this.baseValue,
  ) : _value = baseValue,
      super(propagation) {
    _valueEntry = createValueEntry(baseValue);
  }

  /// The scheduled value at [epoch] (0-based).
  double computeValue(int epoch);

  @override
  double get value => _value;

  @override
  E get valueEntry => _valueEntry;

  @override
  double get initialValue => baseValue;

  @override
  void setValue(double value) {
    _value = value;
    _valueEntry = createValueEntry(value);
  }

  @override
  void initializeValue() {
    _epoch = 0;
    setValue(computeValue(0));
  }

  @override
  void resetValue() {
    _epoch = 0;
    setValue(computeValue(0));
  }

  @override
  void updateValue() {
    _epoch++;
    setValue(computeValue(_epoch));
  }
}

/// Step decay: `base · gamma^(epoch ~/ stepSize)`.
class StepDecayStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends LearningRateScheduleStrategy<N, E, T> {
  final int stepSize;
  final double gamma;

  StepDecayStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation,
    double baseValue, {
    this.stepSize = 100,
    this.gamma = 0.5,
  }) : super(propagation, baseValue);

  @override
  double computeValue(int epoch) => baseValue * pow(gamma, epoch ~/ stepSize);
}

/// Exponential decay: `base · gamma^epoch`.
class ExponentialDecayStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends LearningRateScheduleStrategy<N, E, T> {
  final double gamma;

  ExponentialDecayStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation,
    double baseValue, {
    this.gamma = 0.99,
  }) : super(propagation, baseValue);

  @override
  double computeValue(int epoch) => baseValue * pow(gamma, epoch);
}

/// Cosine annealing from `base` down to [minValue] over [maxEpochs].
class CosineAnnealingStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends LearningRateScheduleStrategy<N, E, T> {
  final int maxEpochs;
  final double minValue;

  CosineAnnealingStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation,
    double baseValue, {
    this.maxEpochs = 1000,
    this.minValue = 0.0,
  }) : super(propagation, baseValue);

  @override
  double computeValue(int epoch) {
    final t = (epoch.clamp(0, maxEpochs)) / maxEpochs;
    return minValue + 0.5 * (baseValue - minValue) * (1 + cos(pi * t));
  }
}

/// Linear warmup to `base` over [warmupEpochs], constant thereafter.
class WarmupStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends LearningRateScheduleStrategy<N, E, T> {
  final int warmupEpochs;

  WarmupStrategy(
    Propagation<N, E, T, dynamic, dynamic> propagation,
    double baseValue, {
    this.warmupEpochs = 50,
  }) : super(propagation, baseValue);

  @override
  double computeValue(int epoch) => epoch >= warmupEpochs
      ? baseValue
      : baseValue * (epoch + 1) / warmupEpochs;
}
