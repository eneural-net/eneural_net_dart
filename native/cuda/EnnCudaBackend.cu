// eneural_net — native CUDA (NVIDIA GPU) backend: BATCHED whole-epoch trainer.
//
// Mirrors the Metal backend (native/macos/metal): all samples are processed at
// once as matrices, so dispatch count is O(layers), independent of the sample
// count. Each epoch is:
//   forward:   NET_{L+1} = OUT_L · W_L
//   backprop:  DELTA_L   = DELTA_{L+1} · W_L^T
//   gradient:  G_L       = OUT_L^T · DELTA_{L+1}     (batch-summed in one GEMM)
// The three GEMMs use cuBLAS (SGEMM); activation/delta/update are the elementwise
// CUDA kernels below, translated 1:1 from the pure-Dart / Metal numerics (incl.
// the bias-row=1 rule) so weights read back match the Dart trainer within
// float32 tolerance.
//
// Activation ids: 0=Linear, 1=Sigmoid, 2=SigmoidFast, 3=SigmoidBoundedFast.
// C symbols are prefixed `enn_cuda_` so this library can coexist with the CPU /
// Metal backends.
//
// ---- cuBLAS layout note -------------------------------------------------------
// cuBLAS is column-major. Our matrices are row-major, so a row-major R[a×b] is
// exactly the column-major matrix [b×a] over the same memory. Under that view:
//   OUT   row [M×in]      == col [in×M]
//   W     row [in×out]    == col [out×in]
//   NET   row [M×out]     == col [out×M]
// which yields the three SGEMM calls in gemmForward / gemmBackprop / gemmGradient
// (each derived and commented at its call site).
// ------------------------------------------------------------------------------

#include "cuda_common.h"

#include <vector>

// ============================================================
// DEVICE MATH — identical formulas to EnnMetalShaders / pure Dart.
// ============================================================

__device__ __forceinline__ float enn_activate(int id, float x, float scale) {
    switch (id) {
        case 0: return x;
        case 1: return 1.0f / (1.0f + __expf(-x));
        case 2: { float x3 = x * 3.0f; return 0.5f + (x3 / (2.5f + fabsf(x3)) / 2.0f); }
        case 3: { float v = fminf(fmaxf(x, -scale), scale) / scale;
                  return 0.5f + (v / (1.0f + v * v)); }
        default: return x;
    }
}

__device__ __forceinline__ float enn_deriv(int id, float o, float flat, bool withFlat) {
    if (id == 0) { return withFlat ? (1.0f + flat) : 1.0f; }
    float d = o * (1.0f - o);
    return withFlat ? (d + flat) : d;
}

__device__ __forceinline__ float enn_sign_zt(float v, float tol) {
    if (v > 0.0f) { return v < tol ? 0.0f : 1.0f; }
    return v > -tol ? 0.0f : -1.0f;
}

// ============================================================
// KERNELS
// ============================================================

__global__ void enn_k_fill(float* buf, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = value;
}

__global__ void enn_k_copy(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Load the whole sample batch into out[0]: real inputs in columns
// [0, inputSize), bias column forced to 1. Grid over (layerSize, numSamples).
__global__ void enn_k_load_input_batch(
    float* out0, const float* inputs,
    int layerSize, int inputSize, int biasIndex, int numSamples
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= layerSize || s >= numSamples) return;
    float v;
    if (c == biasIndex) v = 1.0f;
    else if (c < inputSize) v = inputs[s * inputSize + c];
    else v = 0.0f;
    out0[s * layerSize + c] = v;
}

// out[idx] = activate(net[idx]); bias column forced to 1.
__global__ void enn_k_activate_batch(
    const float* net, float* out,
    int n, int actId, float scale, int biasIndex, int numSamples
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || s >= numSamples) return;
    int idx = s * n + j;
    out[idx] = (j == biasIndex) ? 1.0f : enn_activate(actId, net[idx], scale);
}

// error = target - o; errBuf[idx] = error^2; delta = error * deriv(o).
__global__ void enn_k_output_delta_batch(
    const float* outLast, const float* targets, float* delta, float* errBuf,
    int n, int actId, float flat, int numSamples
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= n || s >= numSamples) return;
    int idx = s * n + k;
    float o = outLast[idx];
    float err = targets[idx] - o;
    errBuf[idx] = err * err;
    delta[idx] = err * enn_deriv(actId, o, flat, true);
}

// Finalize a middle-layer delta in place: the GEMM has written the neuron error
// into `delta`; multiply by the activation derivative of `out`.
__global__ void enn_k_delta_batch(
    float* delta, const float* out,
    int n, int actId, float flat, int withFlat, int numSamples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || s >= numSamples) return;
    int idx = s * n + i;
    delta[idx] = delta[idx] * enn_deriv(actId, out[idx], flat, withFlat != 0);
}

// Backpropagation update: delta = lr*g + momentum*prevDelta; w += delta.
__global__ void enn_k_update_bp(
    float* w, const float* g, float* prevDelta,
    float lr, float momentum, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = lr * g[i] + momentum * prevDelta[i];
    prevDelta[i] = d;
    w[i] += d;
}

// iRProp+ update.
__global__ void enn_k_update_rprop(
    float* w, const float* g, const float* gPrev,
    float* prevDelta, float* lastUpdate,
    float etaPlus, float etaMinus, float dMin, float dMax,
    int backtrack, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float tol = 1e-20f;
    float grad = g[i];
    float pg = gPrev[i];
    float pd = prevDelta[i];
    float change = enn_sign_zt(grad * pg, tol);
    float gs = enn_sign_zt(grad, tol);
    if (pd < 0.0f) { pd = -pd; change = 0.0f; }
    float ud;
    float wu;
    if (change > 0.0f) {
        ud = fminf(pd * etaPlus, dMax);
        wu = gs * ud;
    } else if (change < 0.0f) {
        ud = fmaxf(pd * etaMinus, dMin);
        ud = -ud;
        wu = (backtrack != 0) ? (lastUpdate[i] * -1.0f) : 0.0f;
    } else {
        ud = pd;
        wu = gs * ud;
    }
    prevDelta[i] = ud;
    lastUpdate[i] = wu;
    w[i] += wu;
}

// Single-sample forward pass (used by enn_cuda_activate inference).
__global__ void enn_k_forward(
    const float* w, const float* outPrev, float* outNext,
    int inSize, int outSize, int actId, float scale, int biasIndex
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= outSize) return;
    float acc = 0.0f;
    for (int i = 0; i < inSize; i++) {
        acc += w[i * outSize + j] * outPrev[i];
    }
    float o = enn_activate(actId, acc, scale);
    if (j == biasIndex) o = 1.0f;
    outNext[j] = o;
}

// ============================================================
// RESIDENT NETWORK STATE
// ============================================================

struct EnnCudaNetwork {
    int numLayers = 0;
    std::vector<int> sizes;
    std::vector<int> withBias;      // 0/1
    std::vector<int> activation;
    std::vector<float> actScale;
    std::vector<float> flatSpot;
    std::vector<int> wCounts;

    // Per connection L -> L+1 (device, row-major [inSize][outSize]).
    std::vector<float*> w, g, gPrev, prevDelta, lastUpdate;

    // Single-sample inference buffers (device) + host staging.
    std::vector<float*> outInf;
    std::vector<float> hostStage;   // reused for host<->device copies

    // Batched activation buffers (device, [numSamples x sizes[L]]).
    std::vector<float*> out, net, delta;

    // Samples (resident, device).
    int numSamples = 0, inputSize = 0, outputSize = 0;
    float* inputs = nullptr;
    float* targets = nullptr;
    float* errBuf = nullptr;

    // Update rule.
    bool useRprop = false;
    float rDelta0 = 0.10f, rMin = 1.0e-6f, rMax = 50.0f;
    float rEtaPlus = 1.2f, rEtaMinus = 0.5f;

    float globalLearnError = 1.0f;
    float lastGlobalLearnError = 1.0f;

    cublasHandle_t blas = nullptr;

    int weightsLength() const {
        int t = 0;
        for (int c : wCounts) t += c;
        return t;
    }
};

// ------------------------------------------------------------
// Allocation helpers.
// ------------------------------------------------------------
static float* enn_dev_alloc(int count, float fill) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t)(count > 0 ? count : 1) * sizeof(float)));
    if (fill == 0.0f) {
        CUDA_CHECK(cudaMemset(p, 0, (size_t)(count > 0 ? count : 1) * sizeof(float)));
    } else {
        int block = 256, grid = enn_ceil_div(count, block);
        enn_k_fill<<<grid, block>>>(p, fill, count);
    }
    return p;
}

static void enn_dev_free(std::vector<float*>& v) {
    for (float* p : v) {
        if (p) cudaFree(p);
    }
    v.clear();
}

// Launch geometry helpers.
static inline dim3 grid2d(int gx, int gy) {
    return dim3(enn_ceil_div(gx, 16), enn_ceil_div(gy, 16), 1);
}
static const dim3 block2d(16, 16, 1);

// ------------------------------------------------------------
// cuBLAS GEMM wrappers (row-major semantics via column-major cuBLAS).
// ------------------------------------------------------------

// NET[M×out] = OUT[M×in] · W[in×out].  Col-major: NET_col[out×M] = W_col·OUT_col.
static void gemmForward(EnnCudaNetwork* n, const float* OUT, const float* W,
                        float* NET, int M, int inSize, int outSize) {
    const float a = 1.0f, b = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        n->blas, CUBLAS_OP_N, CUBLAS_OP_N,
        outSize, M, inSize,
        &a, W, outSize, OUT, inSize,
        &b, NET, outSize));
}

// DELTA[M×in] = DELTA_next[M×out] · W^T.  Col-major: DELTA_col[in×M] =
// W_col^T[in×out] · DELTA_next_col[out×M].
static void gemmBackprop(EnnCudaNetwork* n, const float* W, const float* DELTA_next,
                         float* DELTA, int M, int inSize, int outSize) {
    const float a = 1.0f, b = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        n->blas, CUBLAS_OP_T, CUBLAS_OP_N,
        inSize, M, outSize,
        &a, W, outSize, DELTA_next, outSize,
        &b, DELTA, inSize));
}

// G[in×out] = OUT[M×in]^T · DELTA_next[M×out]  (batch-summed).  Col-major:
// G_col[out×in] = DELTA_next_col[out×M] · OUT_col^T[M×in].
static void gemmGradient(EnnCudaNetwork* n, const float* OUT, const float* DELTA_next,
                         float* G, int M, int inSize, int outSize) {
    const float a = 1.0f, b = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        n->blas, CUBLAS_OP_N, CUBLAS_OP_T,
        outSize, inSize, M,
        &a, DELTA_next, outSize, OUT, inSize,
        &b, G, outSize));
}

// ============================================================
// C API — mirrors enn_cpu_* / enn_metal_*.
// ============================================================

ENN_EXPORT int32_t enn_cuda_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0) ? 1 : 0;
}

ENN_EXPORT void* enn_cuda_create_network(
    int32_t numLayers,
    const int32_t* neuronCounts,
    const int32_t* withBias,
    const int32_t* activationIds,
    const float* activationScale,
    const float* flatSpots
) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return nullptr;

    EnnCudaNetwork* n = new EnnCudaNetwork();
    n->numLayers = (int)numLayers;
    n->sizes.resize(n->numLayers);
    n->withBias.resize(n->numLayers);
    n->activation.resize(n->numLayers);
    n->actScale.resize(n->numLayers);
    n->flatSpot.resize(n->numLayers);
    for (int i = 0; i < n->numLayers; i++) {
        n->sizes[i] = (int)neuronCounts[i];
        n->withBias[i] = withBias[i] != 0 ? 1 : 0;
        n->activation[i] = (int)activationIds[i];
        n->actScale[i] = activationScale[i];
        n->flatSpot[i] = flatSpots[i];
    }

    if (cublasCreate(&n->blas) != CUBLAS_STATUS_SUCCESS) {
        delete n;
        return nullptr;
    }

    for (int l = 0; l < n->numLayers - 1; l++) {
        int cnt = n->sizes[l] * n->sizes[l + 1];
        n->wCounts.push_back(cnt);
        n->w.push_back(enn_dev_alloc(cnt, 0.0f));
        n->g.push_back(enn_dev_alloc(cnt, 0.0f));
        n->gPrev.push_back(enn_dev_alloc(cnt, 0.0f));
        n->prevDelta.push_back(enn_dev_alloc(cnt, 0.10f));
        n->lastUpdate.push_back(enn_dev_alloc(cnt, 0.0f));
    }
    for (int l = 0; l < n->numLayers; l++) {
        n->outInf.push_back(enn_dev_alloc(n->sizes[l], 0.0f));
    }
    return (void*)n;
}

ENN_EXPORT void enn_cuda_destroy(void* ptr) {
    if (!ptr) return;
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    enn_dev_free(n->w);
    enn_dev_free(n->g);
    enn_dev_free(n->gPrev);
    enn_dev_free(n->prevDelta);
    enn_dev_free(n->lastUpdate);
    enn_dev_free(n->outInf);
    enn_dev_free(n->out);
    enn_dev_free(n->net);
    enn_dev_free(n->delta);
    if (n->inputs) cudaFree(n->inputs);
    if (n->targets) cudaFree(n->targets);
    if (n->errBuf) cudaFree(n->errBuf);
    if (n->blas) cublasDestroy(n->blas);
    delete n;
}

ENN_EXPORT int32_t enn_cuda_weights_length(void* ptr) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    return (int32_t)n->weightsLength();
}

// Flat order == ANN.allWeights: layers L = (numLayers-2) down to 0.
ENN_EXPORT void enn_cuda_set_weights(void* ptr, const float* weights, int32_t length) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    int offset = 0;
    for (int l = n->numLayers - 2; l >= 0; l--) {
        int cnt = n->wCounts[l];
        CUDA_CHECK(cudaMemcpy(n->w[l], weights + offset,
                              (size_t)cnt * sizeof(float),
                              cudaMemcpyHostToDevice));
        offset += cnt;
    }
}

ENN_EXPORT void enn_cuda_get_weights(void* ptr, float* outWeights, int32_t length) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    int offset = 0;
    for (int l = n->numLayers - 2; l >= 0; l--) {
        int cnt = n->wCounts[l];
        CUDA_CHECK(cudaMemcpy(outWeights + offset, n->w[l],
                              (size_t)cnt * sizeof(float),
                              cudaMemcpyDeviceToHost));
        offset += cnt;
    }
}

ENN_EXPORT void enn_cuda_set_samples(
    void* ptr,
    int32_t numSamples, int32_t inputSize, int32_t outputSize,
    const float* inputs, const float* targets
) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    n->numSamples = (int)numSamples;
    n->inputSize = (int)inputSize;
    n->outputSize = (int)outputSize;
    int inCount = n->numSamples * n->inputSize;
    int outCount = n->numSamples * n->outputSize;

    if (n->inputs) { cudaFree(n->inputs); n->inputs = nullptr; }
    if (n->targets) { cudaFree(n->targets); n->targets = nullptr; }
    if (n->errBuf) { cudaFree(n->errBuf); n->errBuf = nullptr; }

    CUDA_CHECK(cudaMalloc(&n->inputs, (size_t)(inCount > 0 ? inCount : 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n->targets, (size_t)(outCount > 0 ? outCount : 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n->errBuf, (size_t)(outCount > 0 ? outCount : 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(n->inputs, inputs, (size_t)inCount * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(n->targets, targets, (size_t)outCount * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Allocate batched activation buffers.
    enn_dev_free(n->out);
    enn_dev_free(n->net);
    enn_dev_free(n->delta);
    for (int l = 0; l < n->numLayers; l++) {
        n->out.push_back(enn_dev_alloc(n->numSamples * n->sizes[l], 0.0f));
        n->net.push_back(enn_dev_alloc(n->numSamples * n->sizes[l], 0.0f));
        n->delta.push_back(enn_dev_alloc(n->numSamples * n->sizes[l], 0.0f));
    }
}

ENN_EXPORT void enn_cuda_config_backprop(void* ptr) {
    ((EnnCudaNetwork*)ptr)->useRprop = false;
}

ENN_EXPORT void enn_cuda_config_rprop(
    void* ptr,
    float delta0, float deltaMin, float deltaMax, float etaPlus, float etaMinus
) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    n->useRprop = true;
    n->rDelta0 = delta0;
    n->rMin = deltaMin;
    n->rMax = deltaMax;
    n->rEtaPlus = etaPlus;
    n->rEtaMinus = etaMinus;
}

// Mirrors Propagation.reset(): prevDelta -> 0.10, lastUpdate -> 0.
ENN_EXPORT void enn_cuda_reset_optimizer(void* ptr) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    for (int l = 0; l < n->numLayers - 1; l++) {
        int cnt = n->wCounts[l];
        int block = 256, grid = enn_ceil_div(cnt, block);
        enn_k_fill<<<grid, block>>>(n->prevDelta[l], 0.10f, cnt);
        CUDA_CHECK(cudaMemset(n->lastUpdate[l], 0, (size_t)cnt * sizeof(float)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

ENN_EXPORT float enn_cuda_run_epoch(void* ptr, float target, float lr, float momentum) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    int last = n->numLayers - 1;
    int m = n->numSamples;
    if (!n->inputs || !n->errBuf || m <= 0) return n->globalLearnError;

    int errCount = n->numSamples * n->outputSize;

    // errBuf = 0.
    CUDA_CHECK(cudaMemset(n->errBuf, 0, (size_t)errCount * sizeof(float)));

    // resetGradients(): gPrev = g (current becomes previous; the gradient GEMM
    // below overwrites g).
    for (int l = 0; l < n->numLayers - 1; l++) {
        int cnt = n->wCounts[l];
        int block = 256, grid = enn_ceil_div(cnt, block);
        enn_k_copy<<<grid, block>>>(n->gPrev[l], n->g[l], cnt);
    }

    // Load input batch into out[0].
    int biasIndex0 = n->withBias[0] ? n->sizes[0] - 1 : -1;
    enn_k_load_input_batch<<<grid2d(n->sizes[0], m), block2d>>>(
        n->out[0], n->inputs, n->sizes[0], n->inputSize, biasIndex0, m);

    // Forward.
    for (int l = 0; l < n->numLayers - 1; l++) {
        gemmForward(n, n->out[l], n->w[l], n->net[l + 1], m,
                    n->sizes[l], n->sizes[l + 1]);
        bool nextIsOutput = (l + 1) == last;
        int biasIndex = (!nextIsOutput && n->withBias[l + 1]) ? (n->sizes[l + 1] - 1) : -1;
        enn_k_activate_batch<<<grid2d(n->sizes[l + 1], m), block2d>>>(
            n->net[l + 1], n->out[l + 1], n->sizes[l + 1],
            n->activation[l + 1], n->actScale[l + 1], biasIndex, m);
    }

    // Output delta + error.
    enn_k_output_delta_batch<<<grid2d(n->sizes[last], m), block2d>>>(
        n->out[last], n->targets, n->delta[last], n->errBuf,
        n->sizes[last], n->activation[last], n->flatSpot[last], m);

    // Backprop (high -> low): gradient for every layer, delta for hidden layers.
    for (int l = n->numLayers - 2; l >= 0; l--) {
        // G[l] = OUT[l]^T · DELTA[l+1]  (batch-summed).
        gemmGradient(n, n->out[l], n->delta[l + 1], n->g[l], m,
                     n->sizes[l], n->sizes[l + 1]);
        if (l > 0) {
            // neuronError DELTA[l] = DELTA[l+1] · W[l]^T
            gemmBackprop(n, n->w[l], n->delta[l + 1], n->delta[l], m,
                         n->sizes[l], n->sizes[l + 1]);
            enn_k_delta_batch<<<grid2d(n->sizes[l], m), block2d>>>(
                n->delta[l], n->out[l], n->sizes[l],
                n->activation[l], n->flatSpot[l], 1 /* withFlatSpot */, m);
        }
    }

    // Global error = sum(err^2) / (outputSize * numSamples). errBuf holds the
    // squared errors (non-negative), so cublasSasum yields their sum.
    float sumSq = 0.0f;
    CUBLAS_CHECK(cublasSasum(n->blas, errCount, n->errBuf, 1, &sumSq));
    float globalError = sumSq / (float)(n->outputSize * n->numSamples);
    n->lastGlobalLearnError = n->globalLearnError;
    n->globalLearnError = globalError;

    if (globalError < target) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return globalError;
    }

    // Weight update.
    if (n->useRprop) {
        int backtrack = n->globalLearnError > n->lastGlobalLearnError ? 1 : 0;
        for (int li = 0; li < n->numLayers - 1; li++) {
            int cnt = n->wCounts[li];
            int block = 256, grid = enn_ceil_div(cnt, block);
            enn_k_update_rprop<<<grid, block>>>(
                n->w[li], n->g[li], n->gPrev[li], n->prevDelta[li], n->lastUpdate[li],
                n->rEtaPlus, n->rEtaMinus, n->rMin, n->rMax, backtrack, cnt);
        }
    } else {
        for (int li = 0; li < n->numLayers - 1; li++) {
            int cnt = n->wCounts[li];
            int block = 256, grid = enn_ceil_div(cnt, block);
            enn_k_update_bp<<<grid, block>>>(
                n->w[li], n->g[li], n->prevDelta[li], lr, momentum, cnt);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return globalError;
}

ENN_EXPORT void enn_cuda_activate(
    void* ptr,
    const float* input, int32_t inputSize,
    float* output, int32_t outputSize
) {
    EnnCudaNetwork* n = (EnnCudaNetwork*)ptr;
    int last = n->numLayers - 1;

    // Stage the input into outInf[0], forcing the bias row to 1.
    int in0 = n->sizes[0];
    if ((int)n->hostStage.size() < in0) n->hostStage.resize(in0);
    for (int i = 0; i < (int)inputSize && i < in0; i++) n->hostStage[i] = input[i];
    if (n->withBias[0]) n->hostStage[in0 - 1] = 1.0f;
    CUDA_CHECK(cudaMemcpy(n->outInf[0], n->hostStage.data(),
                          (size_t)in0 * sizeof(float), cudaMemcpyHostToDevice));

    for (int l = 0; l < n->numLayers - 1; l++) {
        int inSize = n->sizes[l];
        int outSize = n->sizes[l + 1];
        bool nextIsOutput = (l + 1) == last;
        int biasIndex = (!nextIsOutput && n->withBias[l + 1]) ? (n->sizes[l + 1] - 1) : -1;
        int block = 256, grid = enn_ceil_div(outSize, block);
        enn_k_forward<<<grid, block>>>(
            n->w[l], n->outInf[l], n->outInf[l + 1],
            inSize, outSize, n->activation[l + 1], n->actScale[l + 1], biasIndex);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int outN = n->sizes[last];
    CUDA_CHECK(cudaMemcpy(output, n->outInf[last],
                          (size_t)(outputSize < outN ? outputSize : outN) * sizeof(float),
                          cudaMemcpyDeviceToHost));
}
