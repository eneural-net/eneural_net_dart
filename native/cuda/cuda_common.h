// eneural_net — CUDA backend common header: includes, export macro, error
// checking and small math helpers. Shared by EnnCudaBackend.cu.
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------
// Cross-platform C export (Windows .dll / Linux .so).
// ------------------------------------------------------------
#ifdef _WIN32
  #define ENN_EXPORT extern "C" __declspec(dllexport)
#else
  #define ENN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ------------------------------------------------------------
// Error checking — log but do not abort the host process, so a
// misbehaving GPU degrades to the pure-Dart path instead of killing
// the whole program.
// ------------------------------------------------------------
#define CUDA_CHECK(x)                                                     \
    do {                                                                  \
        cudaError_t _e = (x);                                             \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA ERROR %s:%d -> %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(x)                                                   \
    do {                                                                  \
        cublasStatus_t _s = (x);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS ERROR %s:%d -> %d\n",                 \
                    __FILE__, __LINE__, (int)_s);                         \
        }                                                                 \
    } while (0)

static inline int enn_ceil_div(int a, int b) { return (a + b - 1) / b; }
