import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:eneural_net/eneural_net.dart';

/// Shared helpers for the dataset examples: acceleration-backend selection,
/// dataset download/caching, feature normalization, and accuracy.
typedef ANNF = ANN<double, Float32x4, SignalFloat32x4, Scale<double>>;

/// Parses the acceleration backend from CLI [args] (default `none`).
///
///   dart run <example> [none|auto|cpu|metal]
NativeBackend parseBackend(List<String> args) {
  final a = (args.isNotEmpty ? args.first : 'none').toLowerCase();
  return switch (a) {
    'none' => NativeBackend.none,
    'auto' => NativeBackend.auto,
    'cpu' => NativeBackend.cpu,
    'metal' => NativeBackend.metal,
    _ => throw ArgumentError(
      'Unknown backend "$a" (use: none | auto | cpu | metal)',
    ),
  };
}

/// Downloads [url] into the dataset cache (system temp) as [cacheName],
/// returning the file contents. Cached after the first run.
Future<String> fetchDataset(String url, String cacheName) async {
  final dir = Directory('${Directory.systemTemp.path}/eneural_net_datasets');
  final file = File('${dir.path}/$cacheName');
  if (file.existsSync() && file.lengthSync() > 0) {
    return file.readAsStringSync();
  }
  final client = HttpClient();
  try {
    final response = await (await client.getUrl(Uri.parse(url))).close();
    if (response.statusCode != 200) {
      throw HttpException('HTTP ${response.statusCode} for $url');
    }
    final content = await response.transform(utf8.decoder).join();
    dir.createSync(recursive: true);
    file.writeAsStringSync(content);
    return content;
  } finally {
    client.close();
  }
}

/// Min-max normalizes each column of [rows] to `0..1`, in place.
void normalizeColumns(List<List<double>> rows) {
  if (rows.isEmpty) return;
  final n = rows.first.length;
  final mins = List<double>.filled(n, double.infinity);
  final maxs = List<double>.filled(n, -double.infinity);
  for (final r in rows) {
    for (var i = 0; i < n; ++i) {
      if (r[i] < mins[i]) mins[i] = r[i];
      if (r[i] > maxs[i]) maxs[i] = r[i];
    }
  }
  for (final r in rows) {
    for (var i = 0; i < n; ++i) {
      final range = maxs[i] - mins[i];
      r[i] = range == 0 ? 0.0 : (r[i] - mins[i]) / range;
    }
  }
}

/// Classification accuracy: argmax of the network outputs vs the true label.
double classificationAccuracy(
  ANNF ann,
  List<SampleFloat32x4> samples,
  List<int> labels,
) {
  var correct = 0;
  for (var i = 0; i < samples.length; ++i) {
    ann.activate(samples[i].input);
    final out = ann.outputAsDouble;
    var best = 0;
    for (var k = 1; k < out.length; ++k) {
      if (out[k] > out[best]) best = k;
    }
    if (best == labels[i]) correct++;
  }
  return correct / samples.length;
}
