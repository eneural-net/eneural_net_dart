// Minimal `dart:js_interop` bindings for the subset of the WebGPU API used by
// the trainer. Web-only (imported through the conditional resolver facade).

import 'dart:js_interop';

@JS('navigator')
external Navigator get navigator;

extension type Navigator._(JSObject _) implements JSObject {
  external GPU? get gpu;
}

extension type GPU._(JSObject _) implements JSObject {
  external JSPromise<GPUAdapter?> requestAdapter();
}

extension type GPUAdapter._(JSObject _) implements JSObject {
  external JSPromise<GPUDevice> requestDevice();
}

extension type GPUDevice._(JSObject _) implements JSObject {
  external GPUQueue get queue;
  external GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
  external GPUShaderModule createShaderModule(
    GPUShaderModuleDescriptor descriptor,
  );
  external GPUComputePipeline createComputePipeline(
    GPUComputePipelineDescriptor descriptor,
  );
  external GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
  external GPUCommandEncoder createCommandEncoder();
}

extension type GPUQueue._(JSObject _) implements JSObject {
  external void submit(JSArray<GPUCommandBuffer> commandBuffers);
  external void writeBuffer(GPUBuffer buffer, int bufferOffset, JSAny data);
  external JSPromise<JSAny?> onSubmittedWorkDone();
}

extension type GPUBuffer._(JSObject _) implements JSObject {
  external JSPromise<JSAny?> mapAsync(int mode);
  external JSArrayBuffer getMappedRange();
  external void unmap();
  external void destroy();
}

extension type GPUShaderModule._(JSObject _) implements JSObject {}

extension type GPUComputePipeline._(JSObject _) implements JSObject {
  external GPUBindGroupLayout getBindGroupLayout(int index);
}

extension type GPUBindGroupLayout._(JSObject _) implements JSObject {}

extension type GPUBindGroup._(JSObject _) implements JSObject {}

extension type GPUCommandEncoder._(JSObject _) implements JSObject {
  external GPUComputePassEncoder beginComputePass();
  external void copyBufferToBuffer(
    GPUBuffer source,
    int sourceOffset,
    GPUBuffer destination,
    int destinationOffset,
    int size,
  );
  external GPUCommandBuffer finish();
}

extension type GPUComputePassEncoder._(JSObject _) implements JSObject {
  external void setPipeline(GPUComputePipeline pipeline);
  external void setBindGroup(int index, GPUBindGroup bindGroup);
  external void dispatchWorkgroups(int x, int y, int z);
  external void end();
}

extension type GPUCommandBuffer._(JSObject _) implements JSObject {}

// --- Descriptors (JS object literals) ---

extension type GPUBufferDescriptor._(JSObject _) implements JSObject {
  external factory GPUBufferDescriptor({
    int size,
    int usage,
    bool mappedAtCreation,
  });
}

extension type GPUShaderModuleDescriptor._(JSObject _) implements JSObject {
  external factory GPUShaderModuleDescriptor({String code});
}

extension type GPUProgrammableStage._(JSObject _) implements JSObject {
  external factory GPUProgrammableStage({
    GPUShaderModule module,
    String entryPoint,
  });
}

extension type GPUComputePipelineDescriptor._(JSObject _) implements JSObject {
  external factory GPUComputePipelineDescriptor({
    JSAny layout,
    GPUProgrammableStage compute,
  });
}

extension type GPUBufferBinding._(JSObject _) implements JSObject {
  external factory GPUBufferBinding({GPUBuffer buffer});
}

extension type GPUBindGroupEntry._(JSObject _) implements JSObject {
  external factory GPUBindGroupEntry({int binding, GPUBufferBinding resource});
}

extension type GPUBindGroupDescriptor._(JSObject _) implements JSObject {
  external factory GPUBindGroupDescriptor({
    GPUBindGroupLayout layout,
    JSArray<GPUBindGroupEntry> entries,
  });
}

/// WebGPU usage flags.
class GPUBufferUsage {
  static const int mapRead = 0x0001;
  static const int copySrc = 0x0004;
  static const int copyDst = 0x0008;
  static const int uniform = 0x0040;
  static const int storage = 0x0080;
}

/// WebGPU map modes.
class GPUMapMode {
  static const int read = 0x0001;
}
