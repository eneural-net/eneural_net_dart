import 'dart:math';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';
import 'package:test/test.dart';

void main() {
  group('Scale', () {
    setUp(() {
      print('================================================================');
    });

    test('ScaleInt.ZERO_TO_ONE', () {
      var scale = ScaleInt.ZERO_TO_ONE;
      print(scale);

      expect(scale.normalize(0), equals(0));
      expect(scale.normalize(1), equals(1));
      expect(scale.normalize(2), equals(2));
    });

    test('ScaleDouble.ZERO_TO_ONE', () {
      var scale = ScaleDouble.ZERO_TO_ONE;
      print(scale);

      expect(scale.normalize(0), equals(0));
      expect(scale.normalize(1), equals(1));

      expect(scale.normalize(0.5), equals(0.5));

      expect(scale.normalize(2), equals(2));
    });

    test('ScaleInt(0,100)', () {
      var scale = ScaleInt(0, 100);
      print(scale);

      expect(scale.normalize(0), equals(0));
      expect(scale.normalize(1), equals(0));

      expect(scale.normalize(10), equals(0));
      expect(scale.normalize(20), equals(0));
      expect(scale.normalize(50), equals(0));
      expect(scale.normalize(70), equals(0));
      expect(scale.normalize(100), equals(1));
    });

    test('ScaleDouble(0,100)', () {
      var scale = ScaleDouble(0, 100);
      print(scale);

      expect(scale.normalize(0), equals(0));
      expect(scale.denormalize(0), equals(0));

      expect(scale.normalize(0.5), equals(0.005));
      expect(scale.denormalize(0.005), equals(0.5));

      expect(scale.normalize(1), equals(0.01));
      expect(scale.denormalize(0.01), equals(1));

      expect(scale.normalize(10), equals(0.1));
      expect(scale.denormalize(0.1), equals(10));

      expect(scale.normalize(20), equals(0.20));
      expect(scale.denormalize(0.20), equals(20));

      expect(scale.normalize(50), equals(0.50));
      expect(scale.denormalize(0.50), equals(50));

      expect(scale.normalize(70), equals(0.70));
      expect(scale.denormalize(0.70), equals(70));

      expect(scale.normalize(100), equals(1.0));
      expect(scale.denormalize(1.0), equals(100));

      expect(scale.normalize(200), equals(2.0));
      expect(scale.denormalize(2.0), equals(200));

      expect(scale.normalize(-0.5), equals(-0.005));
      expect(scale.denormalize(-0.005), equals(-0.5));

      expect(scale.normalize(-100), equals(-1.0));
      expect(scale.denormalize(-1.0), equals(-100));

      expect(scale.normalize(-200), equals(-2.0));
      expect(scale.denormalize(-2.0), equals(-200));
    });

    test('ScaleDouble(0,100)', () {
      var scale = ScaleDouble(100, 200);
      print(scale);

      expect(scale.normalize(100), equals(0));
      expect(scale.denormalize(0), equals(100));

      expect(scale.normalize(150), equals(0.50));
      expect(scale.denormalize(0.50), equals(150));

      expect(scale.normalize(200), equals(1));
      expect(scale.denormalize(1), equals(200));

      expect(scale.normalize(50), equals(-0.50));
      expect(scale.denormalize(-0.50), equals(50));

      expect(scale.normalize(0), equals(-1));
      expect(scale.denormalize(-1), equals(0));
    });

    test('ScaleZoomableInt(0,100,10)', () {
      var scale = ScaleZoomableInt(0, 100, 10);
      print(scale);

      expect(scale.normalize(0), equals(0));
      expect(scale.denormalize(0), equals(0));

      expect(scale.normalize(1), equals(0));
      expect(scale.denormalize(0), equals(0));

      expect(scale.normalize(10), equals(1));
      expect(scale.denormalize(1), equals(10));

      expect(scale.normalize(50), equals(5));
      expect(scale.denormalize(5), equals(50));

      expect(scale.normalize(100), equals(10));
      expect(scale.denormalize(10), equals(100));

      expect(scale.normalize(-10), equals(-1));
      expect(scale.denormalize(-1), equals(-10));

      expect(scale.normalize(-50), equals(-5));
      expect(scale.denormalize(-5), equals(-50));
    });

    test('ScaleZoomableDouble(0,100,10)', () {
      var scale = ScaleZoomableDouble(0, 100, 10);

      expect(scale.normalize(0), equals(0));
      expect(scale.denormalize(0), equals(0));

      expect(scale.normalize(0.5), equals(0.005 * 10));
      expect(scale.denormalize(0.005 * 10), equals(0.5));

      expect(scale.normalize(1), equals(0.01 * 10));
      expect(scale.denormalize(0.01 * 10), equals(1));

      expect(scale.normalize(10), equals(0.1 * 10));
      expect(scale.denormalize(0.1 * 10), equals(10));

      expect(scale.normalize(20), equals(0.20 * 10));
      expect(scale.denormalize(0.20 * 10), equals(20));

      expect(scale.normalize(50), equals(0.50 * 10));
      expect(scale.denormalize(0.50 * 10), equals(50));

      expect(scale.normalize(70), equals(0.70 * 10));
      expect(scale.denormalize(0.70 * 10), equals(70));

      expect(scale.normalize(100), equals(1.0 * 10));
      expect(scale.denormalize(1.0 * 10), equals(100));

      expect(scale.normalize(200), equals(2.0 * 10));
      expect(scale.denormalize(2.0 * 10), equals(200));

      expect(scale.normalize(-0.5), equals(-0.005 * 10));
      expect(scale.denormalize(-0.005 * 10), equals(-0.5));

      expect(scale.normalize(-100), equals(-1.0 * 10));
      expect(scale.denormalize(-1.0 * 10), equals(-100));

      expect(scale.normalize(-200), equals(-2.0 * 10));
      expect(scale.denormalize(-2.0 * 10), equals(-200));
    });
  });

  group('Signal', () {
    test('SampleInt32x4', () {
      for (var i = 0; i < 300; ++i) {
        var values = List<int>.generate(i, (i) => i);
        expect(
            SignalInt32x4.from(values),
            predicate<SignalInt32x4>(
                (s) => equalsSignal(s, 4, values.length, values)));
      }
    });

    test('SignalFloat32x4', () {
      for (var i = 0; i < 300; ++i) {
        var values = List<double>.generate(i + 1, (i) => i.toDouble());
        expect(
            SignalFloat32x4.from(values),
            predicate<SignalFloat32x4>(
                (s) => equalsSignal(s, 4, values.length, values)));
      }
    });
  });

  group('Sample', () {
    test('SampleInt32x4', () {
      var scale = ScaleInt.ZERO_TO_ONE;

      var samples = SampleInt32x4.toListFromString([
        '0,0=0',
        '1,0=1',
        '0,1=1',
        '1,1=0',
      ], scale, true);

      expect(samples.length, equals(4));

      expect(
          samples[0],
          equals(
            SampleInt32x4.fromNormalized([0, 0], [0], scale),
          ));

      expect(samples[0].input, equals(SignalInt32x4.from([0, 0])));
      expect(samples[0].input.length, equals(2));
      expect(samples[0].output, equals(SignalInt32x4.from([0])));
      expect(samples[0].output.length, equals(1));

      expect(samples[1].input, equals(SignalInt32x4.from([1, 0])));
      expect(samples[1].output, equals(SignalInt32x4.from([1])));

      expect(samples[2].input, equals(SignalInt32x4.from([0, 1])));
      expect(samples[2].output, equals(SignalInt32x4.from([1])));

      expect(samples[3].input, equals(SignalInt32x4.from([1, 1])));
      expect(samples[3].output, equals(SignalInt32x4.from([0])));
    });

    test('SampleFloat32x4', () {
      var scale = ScaleDouble.ZERO_TO_ONE;

      var samples = SampleFloat32x4.toListFromString([
        '0,0=0',
        '1,0=1',
        '0,1=1',
        '1,1=0',
      ], scale, true);

      expect(samples.length, equals(4));

      expect(
          samples[0],
          equals(
            SampleFloat32x4.fromNormalized([0, 0], [0], scale),
          ));
    });
  });

  group('ActivationFunction', () {
    test('activationFunctionSigmoid', () {
      var af = ActivationFunctionSigmoid().activate;

      showFunction(
          'activationFunctionSigmoid', (n) => af(n.toDouble()), -12, 12, 1);

      var yAt0 = 0.5;
      var yAt1 = 0.2689414213699951;
      expect(af(-1), equals(yAt1));
      expect(af(0), equals(yAt0));
      expect(af(1), equals(yAt0 + (yAt0 - yAt1)));

      expect(af(-2) < 0.12, isTrue);
      expect(af(2) > 0.88, isTrue);

      expect(af(-4) < 0.018, isTrue);
      expect(af(4) > 0.982, isTrue);

      expect(af(-6) < 0.0025, isTrue);
      expect(af(6) > 0.9975, isTrue);

      expect(af(-10) < 0.00005, isTrue);
      expect(af(10) > 0.99995, isTrue);
    });

    test('activationFunctionSigmoid[SIMD]', () {
      var af = (double o) =>
          ActivationFunctionSigmoid().activateEntry(Float32x4.splat(o)).x;

      showFunction('activationFunctionSigmoid[SIMD]', (n) => af(n.toDouble()),
          -12, 12, 1);

      expect(af(-1), inExclusiveRange(0.268941425, 0.268941435));
      expect(af(0), equals(0.50));
      expect(af(1), inExclusiveRange(0.73105857, 0.731058599));
      expect(af(-2) < 0.12, isTrue);
      expect(af(2) > 0.88, isTrue);

      expect(af(-4) < 0.018, isTrue);
      expect(af(4) > 0.982, isTrue);

      expect(af(-6) < 0.0025, isTrue);
      expect(af(6) > 0.9975, isTrue);

      expect(af(-10) < 0.00005, isTrue);
      expect(af(10) > 0.99995, isTrue);
    });

    test('activationFunctionSigmoidFast', () {
      var af = ActivationFunctionSigmoidFast().activate;

      showFunction(
          'activationFunctionSigmoidFast',
          (n) => af(n.toDouble()),
          -12,
          12,
          1,
          (n) => ActivationFunctionSigmoid().activate(n.toDouble()));

      var yAt0 = 0.5;
      var yAt1 = 0.2272727272727273;
      expect(af(-1), equals(yAt1));
      expect(af(0), equals(yAt0));
      expect(af(1), equals(yAt0 + (yAt0 - yAt1)));

      expect(af(-2) < 0.20, isTrue);
      expect(af(2) > 0.80, isTrue);

      expect(af(-4) < 0.10, isTrue);
      expect(af(4) > 0.90, isTrue);

      expect(af(-6) < 0.07, isTrue);
      expect(af(6) > 0.93, isTrue);

      expect(af(-21) <= 0.02, isTrue);
      expect(af(21) >= 0.98, isTrue);

      expect(af(-41) <= 0.01, isTrue);
      expect(af(41) >= 0.99, isTrue);
    });

    test('ActivationFunctionSigmoidBoundedFast', () {
      var af = ActivationFunctionSigmoidBoundedFast(scale: 6).activate;

      showFunction(
          'activationFunctionSigmoidBoundedFast',
          (n) => af(n.toDouble()),
          -12,
          12,
          1,
          (n) => ActivationFunctionSigmoid().activate(n.toDouble()));

      var yAt0 = 0.5;
      var yAt1 = 0.33783783783783783;
      expect(af(-1), equals(yAt1));
      expect(af(0), equals(yAt0));
      expect(af(1), equals(yAt0 + (yAt0 - yAt1)));

      expect(af(-2) == 0.20, isTrue);
      expect(af(2) == 0.8, isTrue);

      expect(af(-4) < 0.04, isTrue);
      expect(af(4) > 0.96, isTrue);

      expect(af(-6) == 0.0, isTrue);
      expect(af(6) == 1.0, isTrue);

      expect(af(-21) == 0.0, isTrue);
      expect(af(21) == 1.0, isTrue);
    });

    test('activationFunctionSigmoidFastInt', () {
      var af = ActivationFunctionSigmoidFastInt100().activate;

      showFunction(
          'activationFunctionSigmoidFastInt',
          (n) => af(n.toInt()),
          -12,
          12,
          1,
          (n) => ActivationFunctionSigmoid().activate(n.toDouble()) * 100);

      expect(af(-100), equals(1));
      expect(af(-10), equals(9));
      expect(af(-1), equals(34));
      expect(af(0), equals(50));
      expect(af(1), equals(66));
      expect(af(10), equals(91));
      expect(af(100), equals(99));
    });
  });
}

void showFunction(String name, Function(num n) f, num min, num max, num step,
    [Function(num n)? f2]) {
  print('FUNCTION: $name');

  var center = (max - min) / 2 + min;

  for (var i = min;
      i <= max;
      i += ((center - i).abs() < step ? step / 2 : step)) {
    var o = f(i);

    if (f2 != null) {
      var o2 = f2(i);
      var diff = o - o2;
      print('  $i -> $o ; $o2 ($diff)');
    } else {
      print('  $i -> $o');
    }
  }
}

bool equalsSignal<N extends num, E, T extends Signal<N, E, T>>(
    T signal, int entryBlockSize, int length, List values) {
  expect(signal.values, equals(values));
  expect(signal.length, equals(length));

  var entriesLength = length ~/ entryBlockSize;
  if (length % 4 != 0) entriesLength++;
  expect(signal.entriesLength, equals(entriesLength));

  var capacity = entriesLength * entryBlockSize;
  expect(signal.capacity, equals(capacity));

  var lastEntryLength = length - max(0, (capacity - entryBlockSize));
  expect(signal.lastEntryLength, equals(lastEntryLength));

  return true;
}
