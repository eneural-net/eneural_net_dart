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
    return _propagation.signalInstance
        .createEntryFullOf(_propagation.signalInstance.toN(value));
  }
}

/// A parameter strategy with a static/constant value.
class StaticParameterStrategy<N extends num, E, T extends Signal<N, E, T>>
    extends ParameterStrategy<N, E, T> {
  double _value;

  StaticParameterStrategy(Propagation<N, E, T, dynamic, dynamic> propagation,
      [this._value = 0.0])
      : super(propagation) {
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
      _valueEntry = _propagation.signalInstance
          .createEntryFullOf(_propagation.signalInstance.toN(value));
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
      Propagation<N, E, T, dynamic, dynamic> propagation,
      {this.minValue = 0.0,
      this.maxValue = 1.0,
      this.zero = 0.0,
      this.multiplier = 1.0})
      : super(propagation);

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

  LearningRateStrategy(Propagation<N, E, T, dynamic, dynamic> propagation,
      {this.multiplier = 1.0})
      : super(propagation);

  double _learningRate = -1;

  @override
  double get value => _learningRate;

  @override
  E get valueEntry => _learningRateEntry;

  late E _learningRateEntry;

  @override
  void setValue(double learningRate) {
    if (_learningRate != learningRate) {
      _learningRate = learningRate;
      _learningRateEntry = createValueEntry(learningRate);
    }
  }

  double _initialValue = 0;

  @override
  double get initialValue => _initialValue;

  @override
  void initializeValue() {
    _initialValue = (1 / _propagation.trainingSamplesSize) * multiplier;
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
    _noLearnNearZeroCount = 0;

    var lastImprovement =
        _propagation.globalLearnError - _propagation.lastGlobalLearnError;

    if (lastImprovement > 0) {
      if (++_noLearnCount % 10 == 0) {
        var learningRate = _learningRate * 0.90;
        learningRate = max(learningRate, _initialValue / 1000);
        setValue(learningRate);
      }
    } else {
      _noLearnCount = 0;

      var lastImprovementRatio =
          lastImprovement / _propagation.lastGlobalLearnError;

      if (lastImprovementRatio > -1.0E-4 && ++_noLearnNearZeroCount % 10 == 0) {
        var learningRate = _learningRate * 1.10;

        if (learningRate > _initialValue) {
          learningRate = _initialValue;
        }

        setValue(learningRate);
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
  void setValue(double momentum) {
    if (_momentum != momentum) {
      _momentum = momentum;
      _momentumEntry = createValueEntry(momentum);
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
