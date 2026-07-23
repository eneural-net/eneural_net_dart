import '../native_backend.dart';
import 'webgpu_accelerator.dart';

/// Non-web stub: WebGPU is never available.
Future<WebGpuAccelerator?> resolveWebGpuAccelerator({
  required NativeNetworkSpec spec,
  required bool rprop,
}) async => null;
