import 'dart:math';

import 'eneural_net_ann.dart';
import 'eneural_net_extension.dart';
import 'eneural_net_sample.dart';
import 'eneural_net_scale.dart';
import 'eneural_net_signal.dart';

/// A builder for [Training] instances.
typedef TrainingBuilder<N extends num, E, T extends Signal<N, E, T>,
        S extends Scale<N>, P extends Sample<N, E, T, S>>
    = Training<N, E, T, S, P> Function(
        ANN<N, E, T, S> ann, SamplesSet<P> samplesSet);

/// Training logger.
typedef TrainingLogger = void Function(
    Training training, String type, String message,
    [dynamic error, StackTrace? stackTrace]);

/// The default [TrainingLogger].
final TrainingLogger DefaultTrainingLogger =
    (training, type, message, [error, stackTrace]) {
  var algorithmName = training.algorithmName;

  print('$algorithmName> [$type] $message');

  if (error != null) {
    print(error);
  }

  if (stackTrace != null) {
    print(stackTrace);
  }
};

/// Base class for training algorithms.
abstract class Training<N extends num, E, T extends Signal<N, E, T>,
    S extends Scale<N>, P extends Sample<N, E, T, S>> {
  /// The training algorithm name.
  final String algorithmName;

  /// The training subject. Defaults to [samplesSet.subject].
  final String subject;

  /// The [ANN] to train.
  final ANN<N, E, T, S> ann;

  /// The samples set for training.
  final SamplesSet<P> samplesSet;

  final TrainingLogger logger;

  Training(this.ann, this.samplesSet, this.algorithmName,
      {String? subject, TrainingLogger? logger})
      : subject = samplesSet.subject,
        logger = logger ?? DefaultTrainingLogger;

  /// If true logging will be enabled.
  bool logEnabled = true;

  /// If true logging of progress will be enabled.
  bool logProgressEnabled = false;

  void logInfo(String message) {
    if (logEnabled) logger(this, 'INFO', message);
  }

  void logProgress(String message) {
    if (logEnabled && logProgressEnabled) logger(this, 'PROGRESS', message);
  }

  void logWarn(String message) {
    if (logEnabled) logger(this, 'WARN', message);
  }

  void logError(String message, [dynamic error, StackTrace? stackTrace]) {
    if (logEnabled) logger(this, 'ERROR', message, error, stackTrace);
  }

  /// Returns the samples of [samplesSet]
  List<P> get samples => samplesSet.samples;

  /// Returns the subject of [samplesSet]
  String get samplesSubject => samplesSet.subject;

  /// Learn the training of [sample]. Called by [train].
  bool learn(List<P> samples, double targetGlobalError);

  List<N>? _bestTrainingWeights;

  double _bestTrainingError = double.maxFinite;

  /// Reset this instance for a future training sessions.
  void reset() {
    _globalError = double.maxFinite;
    _lastGlobalError = double.maxFinite;
    _trainedEpochs = 0;
    _trainingActivations = 0;
    _trainingSamplesSize = 0;
  }

  int _trainingSamplesSize = 0;

  int get trainingSamplesSize => _trainingSamplesSize;

  void initializeTraining() {
    if (_trainingSamplesSize == 0) {
      _trainingSamplesSize = samples.length;
      initializeParameters();
    }
  }

  /// Returns the total number of failed epochs of all the training session.
  /// A call to [reset] won't reset this value.
  int _totalFailedEpochs = 0;

  int get totalFailedEpochs => _totalFailedEpochs;

  DateTime? _startTime;

  /// The start time of the last training session or null if reset.
  DateTime? get startTime => _startTime;

  DateTime? _endTime;

  /// The end time of the last training session or null if not finished yet.
  DateTime? get endTime => _endTime;

  Duration? get elapsedTime => _startTime != null && _endTime != null
      ? Duration(
          microseconds: _endTime!.microsecondsSinceEpoch -
              _startTime!.microsecondsSinceEpoch)
      : null;

  /// Train the [ann] until [targetGlobalError],
  /// with [maxEpochs] per training session and
  /// a [maxRetries] when a training session can't reach the target global error.
  bool trainUntilGlobalError(
      {double? targetGlobalError,
      int epochsBlock = 50,
      int maxEpochs = 1000000,
      double maxEpochsLimitRatio = 3,
      int maxRetries = 5,
      double retryIncreaseMaxEpochsRatio = 1.50,
      Random? random}) {
    initializeTraining();

    targetGlobalError ??= samplesSet.targetGlobalError;

    if (epochsBlock < 1) {
      epochsBlock = 1;
    }

    if (maxEpochs < 1) {
      maxEpochs = 1;
    }

    if (maxRetries < 0) {
      maxRetries = 0;
    }

    if (retryIncreaseMaxEpochsRatio < 1) {
      retryIncreaseMaxEpochsRatio = 1;
    }

    _setStartTime();

    logInfo(
        'Started $algorithmName training session "$subject". { samples: ${samples.length} ; targetGlobalError: $targetGlobalError }');

    var initialMaxEpochs = maxEpochs;

    if (enableSelectInitialANN) {
      selectInitialANN(samples, targetGlobalError, random);
    }

    var errorEvolution = <double>[];
    var errorEvolutionMaxSize = epochsBlock * 10000;
    var doExtraEpochsCount = 0;

    for (var retry = 0; retry <= maxRetries; ++retry) {
      reset();

      while (_globalError > targetGlobalError && _trainedEpochs < maxEpochs) {
        _trainImpl(samples, epochsBlock, targetGlobalError);
        errorEvolution.add(_globalError);
        errorEvolution.ensureMaximumSize(errorEvolutionMaxSize,
            removeExtras: errorEvolutionMaxSize ~/ 10);
      }

      if (_globalError <= targetGlobalError) {
        _setEndTime();
        _logReachedTargetError(targetGlobalError);
        return true;
      }

      if (doExtraEpochsCount < 3) {
        var lastExtraEpochsGlobalError = _globalError;

        for (var extraEpochsI = 1;
            extraEpochsI <= 3 && _globalError <= lastExtraEpochsGlobalError;
            ++extraEpochsI) {
          var doExtraEpochs = false;

          var movingAverage = errorEvolution.movingAverage(epochsBlock);
          if (movingAverage.length > epochsBlock * 10) {
            var mergeBlockSize = max(movingAverage.length ~/ 10, 3);

            var blocks = movingAverage.mergeBlocks(mergeBlockSize);

            var movingAverageTail1Mean = blocks.getReversed(0);
            var movingAverageTail3Mean = blocks.getReversed(1);
            var movingAverageTail6Mean = blocks.getReversed(2);

            if (movingAverageTail1Mean < movingAverageTail3Mean ||
                movingAverageTail1Mean < movingAverageTail6Mean) {
              doExtraEpochs = true;

              logProgress(
                  'Evolving[$extraEpochsI]> $movingAverageTail6Mean -> $movingAverageTail3Mean -> $movingAverageTail1Mean');
            } else {
              logProgress(
                  'NOT Evolving[$extraEpochsI]> $movingAverageTail6Mean -> $movingAverageTail3Mean -> $movingAverageTail1Mean');
            }
          }

          if (doExtraEpochs) {
            lastExtraEpochsGlobalError = _globalError;

            if (extraEpochsI == 1) doExtraEpochsCount++;

            var maxEpochsExtra = (maxEpochs * (extraEpochsI + 1)).toInt();

            logProgress(
                '[$algorithmName/$subject] Extra Epochs> doExtraEpochsCount: $doExtraEpochsCount ; maxEpochs: $maxEpochs -> $maxEpochsExtra');

            while (_globalError > targetGlobalError &&
                _trainedEpochs < maxEpochsExtra) {
              _trainImpl(samples, epochsBlock, targetGlobalError);
              errorEvolution.add(_globalError);
              errorEvolution.ensureMaximumSize(errorEvolutionMaxSize,
                  removeExtras: errorEvolutionMaxSize ~/ 10);
            }

            if (_globalError <= targetGlobalError) {
              _setEndTime();
              _logReachedTargetError(targetGlobalError);
              return true;
            }
          } else {
            break;
          }
        }
      } else {
        doExtraEpochsCount = 0;
      }

      if (retryIncreaseMaxEpochsRatio > 0) {
        maxEpochs = (maxEpochs * retryIncreaseMaxEpochsRatio).toInt();
        if (maxEpochs > initialMaxEpochs * maxEpochsLimitRatio) {
          maxEpochs = (initialMaxEpochs * maxEpochsLimitRatio).toInt();
        }
      }

      _totalFailedEpochs = _trainedEpochs;

      ann.resetWeights(random);
    }

    if (_bestTrainingWeights != null && _bestTrainingError < _globalError) {
      ann.allWeights = _bestTrainingWeights!;
      _globalError = ann.computeSamplesGlobalError(samples);
    }

    _setEndTime();
    logInfo(
        "(FAIL) Training failed! Can't reach target error $targetGlobalError in $_totalTrainedEpochs epochs. Lowest training error: $_globalError");

    return false;
  }

  void _setStartTime() {
    _startTime = DateTime.now();
  }

  void _setEndTime() {
    _endTime = DateTime.now();
  }

  void _logReachedTargetError(double targetGlobalError) {
    logInfo(
        '(OK) Reached target error in $_totalTrainedEpochs epochs (${elapsedTime?.toStringUnit()}). Final error: $_globalError <= $targetGlobalError');
  }

  int _totalTrainedEpochs = 0;

  /// Returns the total number of epochs of all the training session.
  /// A call to [reset] won't reset this value.
  int get totalTrainedEpochs => _totalTrainedEpochs;

  int _trainedEpochs = 0;

  /// Returns the number of epochs of the last training session.
  int get trainedEpochs => _trainedEpochs;

  int _totalTrainingActivations = 0;

  /// Returns the total number of activations of all the training session.
  /// A call to [reset] won't reset this value.
  int get totalTrainingActivations => _totalTrainingActivations;

  int _trainingActivations = 0;

  /// Returns the number of activations of the last training session.
  int get trainingActivations => _trainingActivations;

  double _globalError = double.maxFinite;

  /// Returns the current training global error (set by [train]).
  double get globalError => _globalError;

  /// The last training global error (prior to current [_globalError] value).
  double _lastGlobalError = double.maxFinite;

  double get lastGlobalError => _lastGlobalError;

  /// Initialize training parameters.
  void initializeParameters() {}

  /// Update training parameters.
  void updateParameters() {}

  String get parameters;

  /// Train the [samples] for n [epochs] and returns the last
  /// global error.
  double train(int epochs, double targetGlobalError) {
    initializeTraining();
    return _trainImpl(samples, epochs, targetGlobalError);
  }

  int _trainImplCount = 0;

  double _trainImpl(List<P> samples, int epochs, double targetGlobalError) {
    var samplesLength = samples.length;

    int trainedEpochs;
    for (trainedEpochs = 0; trainedEpochs < epochs; ++trainedEpochs) {
      if (learn(samples, targetGlobalError)) {
        trainedEpochs++;
        break;
      }

      updateParameters();
    }

    _trainedEpochs += trainedEpochs;
    _totalTrainedEpochs += trainedEpochs;

    var activations = trainedEpochs * samplesLength;
    _trainingActivations += activations;
    _totalTrainingActivations += activations;

    var globalError = computeGlobalError(samples);
    _lastGlobalError = _globalError;
    _globalError = globalError;

    checkBestTrainingError(globalError);

    if ((++_trainImplCount) % 200 == 0) {
      var errorChangeRatio =
          (_globalError - _lastGlobalError) / _lastGlobalError;

      logProgress(
          '[$algorithmName/$subject] $_trainedEpochs / $_totalTrainedEpochs> $globalError ($errorChangeRatio) / $targetGlobalError > $parameters');
    }

    return globalError;
  }

  void checkBestTrainingError(double trainingError) {
    if (trainingError < _bestTrainingError) {
      _bestTrainingError = trainingError;
      _bestTrainingWeights = ann.allWeights;
    }
  }

  static final Random _random = Random();

  /// The initial ANN pool size.
  int initialAnnPoolSize = 100;

  /// Number of epochs to perform in the ANNs in the selection pool.
  int initialAnnEpochs = 6;

  /// If true will select the initial ANN calling [selectInitialANN].
  bool enableSelectInitialANN = true;

  int _selectInitialANNEpochs = 0;

  /// Selects the initial ANN.
  ///
  /// Default implementations uses a pool of random ANNs and selects the
  /// one with the lowest error. The actual implementation only allocates
  /// 1 ANN.
  void selectInitialANN(List<P> samples, double targetGlobalError,
      [Random? random]) {
    if (initialAnnPoolSize <= 1) return;

    random ??= _random;

    var pool = <double, List<N>>{};

    for (var i = 0; i < initialAnnPoolSize; ++i) {
      _trainImpl(samples, initialAnnEpochs, 0.0);

      var samplesErrors = ann.computeSamplesErrors(samples);
      var errorsStatistics = samplesErrors.statistics;

      var errorMean = errorsStatistics.mean;

      if (errorMean <= targetGlobalError) {
        return;
      }

      var error = (errorsStatistics.center + errorsStatistics.mean) / 2;

      pool[error] = ann.allWeights;

      ann.resetWeights(random);
    }

    _selectInitialANNEpochs = _trainedEpochs;

    var allErrors = pool.keys.toList();
    allErrors.sort();

    var minError = allErrors.first;

    var bestWeights = pool[minError]!;

    ann.allWeights = bestWeights;
    _globalError = ann.computeSamplesGlobalError(samples);

    logInfo(
        'Selected initial ANN from poll of size ${pool.length}, executing $_selectInitialANNEpochs epochs. Lowest error: $minError ($_globalError)');
  }

  double computeGlobalError(List<P> samples) {
    return ann.computeSamplesGlobalError(samples);
  }

  @override
  String toString() {
    return '$runtimeType{name: $algorithmName}';
  }
}
