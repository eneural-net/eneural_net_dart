/// Resolves a [NativeAccelerator] for the requested backend.
///
/// This facade selects the real `dart:ffi` implementation on native platforms
/// and a no-op stub on the web, so the package compiles for all targets and the
/// pure-Dart SIMD path remains the default fallback.
export 'native_accelerator_stub.dart'
    if (dart.library.io) 'native_accelerator_io.dart';
