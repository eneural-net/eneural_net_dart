# eneural_net — CUDA (NVIDIA GPU) backend

A native, resident **whole-epoch** trainer for NVIDIA GPUs, mirroring the
[Metal backend](../macos/metal). It accelerates `NativeBackpropagation` /
`NativeRProp` on Windows and Linux hosts with an NVIDIA GPU.

## Design

All samples are processed at once as matrices, so the per-epoch dispatch count is
`O(layers)`, independent of the sample count. Each epoch is three GEMMs per layer
(via **cuBLAS** `SGEMM`) plus a handful of elementwise CUDA kernels:

```
forward:   NET_{L+1} = OUT_L · W_L
backprop:  DELTA_L   = DELTA_{L+1} · W_L^T
gradient:  G_L       = OUT_L^T · DELTA_{L+1}     (batch-summed in one GEMM)
```

The kernels in `EnnCudaBackend.cu` are translated 1:1 from the pure-Dart / Metal
numerics — activation/derivative formulas, the bias-row = 1 rule, and the
Backpropagation and iRProp+ weight updates — so weights read back match the Dart
trainer within float32 tolerance. Weights use the flat `ANN.allWeights` order
(layers `numLayers-2 … 0`). The C symbols are prefixed `enn_cuda_` so this library
coexists with the CPU/Metal backends.

## Requirements

- NVIDIA GPU (compute capability 7.0+ / Volta or newer by default).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (`nvcc` + cuBLAS) on `PATH`.

## Build

**Windows** (Developer Command Prompt / `nvcc` on `PATH`):

```bat
build.bat
```

**Linux:**

```bash
./build.sh
```

**CMake** (either platform):

```bash
cmake -S . -B build && cmake --build build --config Release
```

Each produces `build/libeneural_cuda_<arch>.{dll,so}` (`<arch>` = `x86_64` or
`arm64`). The Dart loader (`lib/src/native/native_accelerator_io.dart`) searches
the project root and `native/cuda/build/` for that file. Request it from Dart with
`NativeBackend.cuda` (or `NativeBackend.auto` on a non-macOS host):

```dart
final trainer = NativeRProp(ann, samples, backend: NativeBackend.cuda);
```

If the library or a CUDA device is missing, resolution returns `null` and training
transparently falls back to the pure-Dart SIMD path.
