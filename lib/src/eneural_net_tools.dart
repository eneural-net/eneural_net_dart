/// A Chronometer useful for benchmarks.
class Chronometer {
  /// The name/title of this chronometer.
  String name;

  Chronometer([this.name = 'Chronometer']);

  DateTime? _startTime;

  /// The start [DateTime] of this chronometer.
  DateTime? get startTime => _startTime;

  /// Starts the chronometer.
  Chronometer start() {
    _startTime = DateTime.now();
    return this;
  }

  DateTime? _stopTime;

  /// The stop [DateTime] of this chronometer.
  DateTime? get stopTime => _stopTime;

  /// Stops the chronometer.
  Chronometer stop({int? operations}) {
    _stopTime = DateTime.now();
    if (operations != null) {
      this.operations = operations;
    }
    return this;
  }

  /// Elapsed time in milliseconds ([stopTime] - [startTime]).
  int get elapsedTimeMs => (_stopTime == null || _startTime == null)
      ? 0
      : (_stopTime!.millisecondsSinceEpoch -
          _startTime!.millisecondsSinceEpoch);

  /// Elapsed time in seconds ([stopTime] - [startTime]).
  double get elapsedTimeSec => elapsedTimeMs / 1000;

  /// Elapsed time ([stopTime] - [startTime]).
  Duration get elapsedTime => Duration(milliseconds: elapsedTimeMs);

  /// Operations performed while this chronometer was running.
  /// Used to compute [hertz].
  int operations = 0;

  /// Returns the [operations] hertz:
  /// The average operations per second of
  /// the period ([elapsedTimeSec]) of this chronometer.
  double get hertz => computeHertz(operations);

  /// Computes hertz for n [operations].
  double computeHertz(int operations) {
    return operations / elapsedTimeSec;
  }

  /// Resets this chronometer for a future [start] and [stop].
  void reset() {
    _startTime = null;
    _stopTime = null;
    operations = 0;
  }

  /// Returns a [String] with information of this chronometer:
  /// Example:
  /// ```
  ///   Backpropagation{elapsedTime: 2955 ms, hertz: 2030456.8527918782 Hz, ops: 6000000, startTime: 2021-04-30 22:16:54.437147, stopTime: 2021-04-30 22:16:57.392758}
  /// ```
  @override
  String toString() {
    return '$name{elapsedTime: $elapsedTimeMs ms, hertz: $hertz Hz, ops: $operations, startTime: $_startTime, stopTime: $_stopTime}';
  }
}
