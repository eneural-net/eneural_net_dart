/// Resolves a [WebGpuAccelerator] when running on a browser with WebGPU.
///
/// Selects the real `dart:js_interop` implementation on the web and a no-op stub
/// elsewhere, so the package compiles for the Dart VM/native and the WebGpu
/// trainers fall back to the pure-Dart path off the web.
export 'webgpu_resolver_stub.dart'
    if (dart.library.js_interop) 'webgpu_resolver_web.dart';
