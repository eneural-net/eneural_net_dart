class ScaleInt extends Scale<int> {
  static final ScaleInt ZERO_TO_ONE = ScaleInt(0, 1);

  @override
  final int zero = 0;

  ScaleInt(int minValue, int maxValue) : super(minValue, maxValue);

  @override
  int toN(num value) => value.toInt();

  @override
  int normalize(int value) => (value - minValue) ~/ range;

  @override
  int normalizeNum(num value) => (value - minValue) ~/ range;

  @override
  int denormalize(int normalizedValue) => (normalizedValue * range) + minValue;

  @override
  String toString() {
    return 'ScaleInt{$minValue .. $maxValue}';
  }
}

class ScaleDouble extends Scale<double> {
  static final ScaleDouble ZERO_TO_ONE = ScaleDouble(0, 1);

  @override
  final double zero = 0;

  ScaleDouble(double minValue, double maxValue) : super(minValue, maxValue);

  @override
  double toN(num value) => value.toDouble();

  @override
  double normalize(double value) => (value - minValue) / range;

  @override
  double normalizeNum(num value) => (value - minValue) / range;

  @override
  double denormalize(double normalizedValue) =>
      (normalizedValue * range) + minValue;

  @override
  String toString() {
    return 'ScaleDouble{$minValue .. $maxValue}';
  }
}

abstract class Scale<N extends num> {
  final N minValue;

  final N maxValue;

  late final N range;

  Scale(this.minValue, this.maxValue) {
    if (maxValue <= minValue) {
      throw ArgumentError('Invalid scale> min:$minValue .. max:$maxValue');
    }
    range = toN((maxValue - minValue));
  }

  N get zero;

  N toN(num value);

  N normalize(N value);

  N normalizeNum(num value);

  N denormalize(N normalizedValue);

  List<N> normalizeList(List<N> values) =>
      values.map((e) => normalize(e)).toList();

  List<N> denormalizeList(List<N> normalizedValues) =>
      normalizedValues.map((e) => denormalize(e)).toList();

  @override
  String toString();

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Scale &&
          runtimeType == other.runtimeType &&
          minValue == other.minValue &&
          maxValue == other.maxValue;

  @override
  int get hashCode => minValue.hashCode ^ maxValue.hashCode;
}

class ScaleZoomableInt extends ScaleZoomable<int> {
  late final int rangeZoomed;

  @override
  final int zero = 0;

  ScaleZoomableInt(int minValue, int maxValue, int zoom)
      : super(minValue, maxValue, zoom) {
    rangeZoomed = range ~/ zoom;
  }

  @override
  int toN(num value) => value.toInt();

  @override
  int normalize(int value) => (value - minValue) ~/ rangeZoomed;

  @override
  int normalizeNum(num value) => (value - minValue) ~/ rangeZoomed;

  @override
  int denormalize(int normalizedValue) =>
      (normalizedValue * rangeZoomed) + minValue;

  @override
  String toString() {
    return 'ScaleZoomableInt{$minValue .. $maxValue * $zoom}';
  }
}

class ScaleZoomableDouble extends ScaleZoomable<double> {
  late final double rangeZoomed;

  @override
  final double zero = 0;

  ScaleZoomableDouble(double minValue, double maxValue, double zoom)
      : super(minValue, maxValue, zoom) {
    rangeZoomed = range / zoom;
  }

  @override
  double toN(num value) => value.toDouble();

  @override
  double normalize(double value) => (value - minValue) / rangeZoomed;

  @override
  double normalizeNum(num value) => (value - minValue) / rangeZoomed;

  @override
  double denormalize(double normalizedValue) =>
      (normalizedValue * rangeZoomed) + minValue;

  @override
  String toString() {
    return 'ScaleZoomableDouble{$minValue .. $maxValue * $zoom}';
  }
}

abstract class ScaleZoomable<N extends num> extends Scale<N> {
  final N zoom;

  ScaleZoomable(N minValue, N maxValue, this.zoom) : super(minValue, maxValue);
}
