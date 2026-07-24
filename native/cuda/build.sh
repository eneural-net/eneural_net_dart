#!/usr/bin/env bash
# Build the eneural_net CUDA backend on Linux into build/.
# Requires the NVIDIA CUDA Toolkit (nvcc + cuBLAS) on PATH.
set -euo pipefail

cd "$(dirname "$0")"
rm -rf build
mkdir -p build

case "$(uname -m)" in
  x86_64|amd64) ARCH=x86_64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

nvcc -O3 --shared --compiler-options -fPIC --use_fast_math -lcublas \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -I. \
    -o "build/libeneural_cuda_${ARCH}.so" \
    EnnCudaBackend.cu

echo "Build succeeded: build/libeneural_cuda_${ARCH}.so"
