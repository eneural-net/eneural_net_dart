class Chronometer {
  String name;

  Chronometer([this.name = 'Chronometer']);

  DateTime? startTime;

  Chronometer start() {
    startTime = DateTime.now();
    return this;
  }

  DateTime? stopTime;

  Chronometer stop() {
    stopTime = DateTime.now();
    return this;
  }

  int get elapsedTimeMs => (stopTime == null || startTime == null)
      ? 0
      : (stopTime!.millisecondsSinceEpoch - startTime!.millisecondsSinceEpoch);

  double get elapsedTimeSec => elapsedTimeMs / 1000;

  Duration get elapsedTime => Duration(milliseconds: elapsedTimeMs);

  int operations = 0;

  double get hertz => computeHertz(operations);

  double computeHertz(int operations) {
    return operations / elapsedTimeSec;
  }

  void reset() {
    startTime = null;
    stopTime = null;
    operations = 0;
  }

  @override
  String toString() {
    return '$name{elapsedTime: $elapsedTimeMs ms, hertz: $hertz Hz, ops: $operations, startTime: $startTime, stopTime: $stopTime}';
  }
}
