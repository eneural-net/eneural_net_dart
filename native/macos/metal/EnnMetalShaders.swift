// Metal Shading Language kernels for the eneural_net whole-epoch trainer.
//
// The trainer is BATCHED: all samples are processed at once as matrices. Each
// epoch is a handful of GEMMs (via MetalPerformanceShaders) plus the elementwise
// kernels below — dispatch count is O(layers), independent of the sample count.
// Buffers holding per-sample activations are laid out row-major, sample-major:
// index = sample * layerSize + feature.
//
// The kernels reproduce the pure-Dart numerics exactly (activation/derivative
// formulas, the bias-row=1 rule, Backpropagation and iRProp+ updates). They are
// compiled at runtime from this string (no precompiled .metallib).

enum EnnMetalShaders {
    static let source = """
#include <metal_stdlib>
using namespace metal;

inline float enn_activate(int id, float x, float scale) {
    switch (id) {
        case 0: return x;
        case 1: return 1.0 / (1.0 + exp(-x));
        case 2: { float x3 = x * 3.0; return 0.5 + (x3 / (2.5 + fabs(x3)) / 2.0); }
        case 3: { float v = clamp(x, -scale, scale) / scale; return 0.5 + (v / (1.0 + v * v)); }
        default: return x;
    }
}

inline float enn_deriv(int id, float o, float flat, bool withFlat) {
    if (id == 0) { return withFlat ? (1.0 + flat) : 1.0; }
    float d = o * (1.0 - o);
    return withFlat ? (d + flat) : d;
}

inline float enn_sign_zt(float v, float tol) {
    if (v > 0) { return v < tol ? 0.0 : 1.0; }
    return v > -tol ? 0.0 : -1.0;
}

kernel void enn_k_zero(
    device float* buf [[buffer(0)]],
    constant int& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i < n) buf[i] = 0.0;
}

kernel void enn_k_copy(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i < n) dst[i] = src[i];
}

// Load the whole sample batch into out[0]: real inputs in columns
// [0, inputSize), bias column forced to 1. Grid = (layerSize, numSamples).
kernel void enn_k_load_input_batch(
    device float* out0 [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    constant int& layerSize [[buffer(2)]],
    constant int& inputSize [[buffer(3)]],
    constant int& biasIndex [[buffer(4)]],
    constant int& numSamples [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int c = (int)gid.x;
    int s = (int)gid.y;
    if (c >= layerSize || s >= numSamples) return;
    float v;
    if (c == biasIndex) v = 1.0;
    else if (c < inputSize) v = inputs[s * inputSize + c];
    else v = 0.0;
    out0[s * layerSize + c] = v;
}

// out[idx] = activate(net[idx]) ; bias column forced to 1.
// Grid = (N, numSamples).
kernel void enn_k_activate_batch(
    device const float* net [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& actId [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& biasIndex [[buffer(5)]],
    constant int& numSamples [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int j = (int)gid.x;
    int s = (int)gid.y;
    if (j >= n || s >= numSamples) return;
    int idx = s * n + j;
    out[idx] = (j == biasIndex) ? 1.0 : enn_activate(actId, net[idx], scale);
}

// error = target - o ; errBuf[idx] = error^2 ; delta = error * deriv(o).
// Grid = (N, numSamples).
kernel void enn_k_output_delta_batch(
    device const float* outLast [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* delta [[buffer(2)]],
    device float* errBuf [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant int& actId [[buffer(5)]],
    constant float& flat [[buffer(6)]],
    constant int& numSamples [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int k = (int)gid.x;
    int s = (int)gid.y;
    if (k >= n || s >= numSamples) return;
    int idx = s * n + k;
    float o = outLast[idx];
    float err = targets[idx] - o;
    errBuf[idx] = err * err;
    delta[idx] = err * enn_deriv(actId, o, flat, true);
}

// Finalize the middle-layer delta in place: the GEMM has written the neuron
// error into `delta`; multiply by the activation derivative of `out`.
// Grid = (N, numSamples).
kernel void enn_k_delta_batch(
    device float* delta [[buffer(0)]],
    device const float* out [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& actId [[buffer(3)]],
    constant float& flat [[buffer(4)]],
    constant int& withFlat [[buffer(5)]],
    constant int& numSamples [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int i = (int)gid.x;
    int s = (int)gid.y;
    if (i >= n || s >= numSamples) return;
    int idx = s * n + i;
    delta[idx] = delta[idx] * enn_deriv(actId, out[idx], flat, withFlat != 0);
}

// Backpropagation update: delta = lr*g + momentum*prevDelta ; w += delta
kernel void enn_k_update_bp(
    device float* w [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float* prevDelta [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    constant int& n [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= n) return;
    float d = lr * g[i] + momentum * prevDelta[i];
    prevDelta[i] = d;
    w[i] += d;
}

// iRProp+ update.
kernel void enn_k_update_rprop(
    device float* w [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device const float* gPrev [[buffer(2)]],
    device float* prevDelta [[buffer(3)]],
    device float* lastUpdate [[buffer(4)]],
    constant float& etaPlus [[buffer(5)]],
    constant float& etaMinus [[buffer(6)]],
    constant float& dMin [[buffer(7)]],
    constant float& dMax [[buffer(8)]],
    constant int& backtrack [[buffer(9)]],
    constant int& n [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
    if ((int)i >= n) return;
    float tol = 1e-20;
    float grad = g[i];
    float pg = gPrev[i];
    float pd = prevDelta[i];
    float change = enn_sign_zt(grad * pg, tol);
    float gs = enn_sign_zt(grad, tol);
    if (pd < 0) { pd = -pd; change = 0.0; }
    float ud;
    float wu;
    if (change > 0) {
        ud = min(pd * etaPlus, dMax);
        wu = gs * ud;
    } else if (change < 0) {
        ud = max(pd * etaMinus, dMin);
        ud = -ud;
        wu = (backtrack != 0) ? (lastUpdate[i] * -1.0) : 0.0;
    } else {
        ud = pd;
        wu = gs * ud;
    }
    prevDelta[i] = ud;
    lastUpdate[i] = wu;
    w[i] += wu;
}

// Single-sample forward pass (used by enn_metal_activate inference).
kernel void enn_k_forward(
    device const float* w [[buffer(0)]],
    device const float* outPrev [[buffer(1)]],
    device float* outNext [[buffer(2)]],
    constant int& inSize [[buffer(3)]],
    constant int& outSize [[buffer(4)]],
    constant int& actId [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant int& biasIndex [[buffer(7)]],
    uint j [[thread_position_in_grid]]
) {
    if ((int)j >= outSize) return;
    float acc = 0.0;
    for (int i = 0; i < inSize; i++) {
        acc += w[i * outSize + (int)j] * outPrev[i];
    }
    float o = enn_activate(actId, acc, scale);
    if ((int)j == biasIndex) o = 1.0;
    outNext[j] = o;
}
"""
}
