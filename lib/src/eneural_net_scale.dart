import 'package:eneural_net/eneural_net.dart';
import 'package:swiss_knife/swiss_knife.dart';

/// A `Scale<int>`.
class ScaleInt extends Scale<int> {
  static final ScaleInt ZERO_TO_ONE = ScaleInt(0, 1);

  @override
  final int zero = 0;

  ScaleInt(int minValue, int maxValue) : super(minValue, maxValue);

  @override
  String get format => 'int';

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

/// A `Scale<double>`.
class ScaleDouble extends Scale<double> {
  static final ScaleDouble ZERO_TO_ONE = ScaleDouble(0, 1);

  @override
  final double zero = 0;

  ScaleDouble(double minValue, double maxValue) : super(minValue, maxValue);

  @override
  String get format => 'double';

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

/// Base class for scales used for [ANN], [Signal] and [Sample].
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

  /// The data format of this scale.
  String get format;

  /// The `zero` value for this scale format.
  N get zero;

  /// Converts [value] to [N].
  N toN(num value);

  /// Normalize [value] to this scale (in the range [0..1])
  N normalize(N value);

  /// Normalize [value] to this scale (in the range [0..1])
  N normalizeNum(num value);

  /// Denormalize [normalizedValue] to values of this scale (in the range [minValue] to [maxValue]).
  N denormalize(N normalizedValue);

  /// Normalizes [values] using [normalize].
  List<N> normalizeList(List<N> values) =>
      values.map((e) => normalize(e)).toList();

  /// Denormalizes [values] using [denormalize].
  List<N> denormalizeList(List<N> normalizedValues) =>
      normalizedValues.map((e) => denormalize(e)).toList();

  @override
  String toString();

  /// Returns `true` if [other] is equals to `this`.
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Scale &&
          runtimeType == other.runtimeType &&
          minValue == other.minValue &&
          maxValue == other.maxValue;

  @override
  int get hashCode => minValue.hashCode ^ maxValue.hashCode;

  /// Converts to an encoded JSON.
  String toJson({bool withIndent = false}) =>
      encodeJSON(toJsonMap(), withIndent: withIndent);

  /// Converts this Scale to a JSON [Map].
  Map<String, dynamic> toJsonMap() => <String, dynamic>{
        'format': format,
        'min': minValue,
        'max': maxValue,
      };

  /// Instantiates a [Scale] from [json].
  factory Scale.fromJson(dynamic json) {
    Map<String, dynamic> jsonMap = json is String ? parseJSON(json) : json;

    var format = jsonMap['format']! as String;

    var min = jsonMap['min']! as num;
    var max = jsonMap['max']! as num;

    switch (format) {
      case 'double':
        {
          return ScaleDouble(min.toDouble(), max.toDouble()) as Scale<N>;
        }
      case 'int':
        {
          return ScaleInt(min.toInt(), max.toInt()) as Scale<N>;
        }
      case 'ZoomableDouble':
        {
          var zoom = jsonMap['zoom']! as num;
          return ScaleZoomableDouble(
              min.toDouble(), max.toDouble(), zoom.toDouble()) as Scale<N>;
        }
      case 'ScaleZoomableInt':
        {
          var zoom = jsonMap['zoom']! as num;
          return ScaleZoomableInt(min.toInt(), max.toInt(), zoom.toInt())
              as Scale<N>;
        }
      default:
        throw StateError('Unknown format: $format');
    }
  }
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
  String get format => 'ZoomableInt';

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
  String get format => 'ZoomableDouble';

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

  @override
  Map<String, dynamic> toJsonMap() {
    var json = super.toJsonMap();
    json['zoom'] = zoom;
    return json;
  }
}

abstract class ScaleZoomable<N extends num> extends Scale<N> {
  final N zoom;

  ScaleZoomable(N minValue, N maxValue, this.zoom) : super(minValue, maxValue);
}
