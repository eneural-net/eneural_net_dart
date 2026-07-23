// eneural_net — native CPU backend (Apple Accelerate / BLAS).
//
// Whole-epoch-on-device trainer: the network topology, weights, optimizer
// state and the full training sample set live in native memory. `enn_run_epoch`
// runs one full training epoch (forward + backprop + weight update over every
// sample) natively, using `cblas_sgemv`/`cblas_sger` for the matrix kernels.
//
// The numeric algorithms mirror the pure-Dart implementation exactly (see
// lib/src/eneural_net_training_propagation.dart, _backpropagation.dart,
// _rprop.dart and eneural_net_ann.dart) so that weights read back from native
// match the Dart trainer within float32 tolerance.
//
// Activation ids: 0=Linear, 1=Sigmoid, 2=SigmoidFast, 3=SigmoidBoundedFast.

import Foundation
import Accelerate

// ============================================================
// Network state
// ============================================================

final class EnnNetwork {
    let numLayers: Int
    let sizes: [Int]          // neuron count per layer (INCLUDING bias slot)
    let withBias: [Bool]      // per layer
    let activation: [Int32]   // per layer activation id
    let actScale: [Float]     // per layer (SigmoidBoundedFast scale)
    let flatSpot: [Float]     // per layer

    // Per connection L -> L+1 (L in 0..<numLayers-1), row-major [inSize][outSize]
    // (source-major, destination-inner) — matches ANN.allWeights row/col order.
    var w: [[Float]]          // weights
    var g: [[Float]]          // current-epoch accumulated gradients
    var gPrev: [[Float]]      // previous-epoch gradients
    var prevDelta: [[Float]]  // previousUpdateDeltas (signed for RProp), init 0.10
    var lastUpdate: [[Float]] // last applied weight update

    // Per layer activation buffers
    var out: [[Float]]        // activated neuron outputs (bias slot forced to 1)
    var net: [[Float]]        // pre-activation nets
    var delta: [[Float]]      // gradient deltas per neuron

    // Samples (resident)
    var numSamples: Int = 0
    var inputSize: Int = 0
    var outputSize: Int = 0
    var inputs: [Float] = []  // [numSamples * inputSize]
    var targets: [Float] = [] // [numSamples * outputSize]

    // Update rule
    var useRprop: Bool = false
    var rDelta0: Float = 0.10
    var rMin: Float = 1.0e-6
    var rMax: Float = 50.0
    var rEtaPlus: Float = 1.2
    var rEtaMinus: Float = 0.5

    // RProp error tracking (persists across epochs), mirrors Propagation.
    var globalLearnError: Float = 1.0
    var lastGlobalLearnError: Float = 1.0

    init(sizes: [Int], withBias: [Bool], activation: [Int32],
         actScale: [Float], flatSpot: [Float]) {
        self.numLayers = sizes.count
        self.sizes = sizes
        self.withBias = withBias
        self.activation = activation
        self.actScale = actScale
        self.flatSpot = flatSpot

        w = []; g = []; gPrev = []; prevDelta = []; lastUpdate = []
        for l in 0..<(numLayers - 1) {
            let n = sizes[l] * sizes[l + 1]
            w.append([Float](repeating: 0, count: n))
            g.append([Float](repeating: 0, count: n))
            gPrev.append([Float](repeating: 0, count: n))
            prevDelta.append([Float](repeating: 0.10, count: n))
            lastUpdate.append([Float](repeating: 0, count: n))
        }

        out = sizes.map { [Float](repeating: 0, count: $0) }
        net = sizes.map { [Float](repeating: 0, count: $0) }
        delta = sizes.map { [Float](repeating: 0, count: $0) }
    }

    var weightsLength: Int {
        var total = 0
        for l in 0..<(numLayers - 1) { total += w[l].count }
        return total
    }
}

// ============================================================
// Activation
// ============================================================

@inline(__always)
private func activate(_ id: Int32, _ x: Float, _ scale: Float) -> Float {
    switch id {
    case 0: // Linear
        return x
    case 1: // Sigmoid
        return 1.0 / (1.0 + expf(-x))
    case 2: // SigmoidFast
        let x3 = x * 3.0
        return 0.5 + (x3 / (2.5 + abs(x3)) / 2.0)
    case 3: // SigmoidBoundedFast
        var v = x
        if v < -scale { v = -scale } else if v > scale { v = scale }
        v = v / scale
        return 0.5 + (v / (1.0 + v * v))
    default:
        return x
    }
}

// Derivative on the activated output `o`.
@inline(__always)
private func derivative(_ id: Int32, _ o: Float, _ flatSpot: Float,
                        _ withFlatSpot: Bool) -> Float {
    switch id {
    case 0: // Linear
        return withFlatSpot ? (1.0 + flatSpot) : 1.0
    default: // Sigmoid family (all share o*(1-o))
        let d = o * (1.0 - o)
        return withFlatSpot ? (d + flatSpot) : d
    }
}

@inline(__always)
private func signZeroTolerance(_ v: Float, _ tol: Float) -> Float {
    if v > 0 { return v < tol ? 0 : 1 }
    return v > -tol ? 0 : -1
}

// ============================================================
// Forward pass (single sample), writing into net/out buffers.
// `out[L]` bias slot is forced to 1 (matches LayerInput/Hidden.activateLayer).
// ============================================================

private func forwardSample(_ n: EnnNetwork, _ sampleIndex: Int) {
    // Input layer.
    let inSize0 = n.inputSize
    for i in 0..<inSize0 {
        n.out[0][i] = n.inputs[sampleIndex * inSize0 + i]
    }
    if n.withBias[0] {
        n.out[0][n.sizes[0] - 1] = 1.0
    }

    for l in 0..<(n.numLayers - 1) {
        let inSize = n.sizes[l]
        let outSize = n.sizes[l + 1]
        let nextIsOutput = (l + 1) == (n.numLayers - 1)

        // net[l+1] = W[l]^T * out[l]  (W row-major [inSize x outSize]).
        n.w[l].withUnsafeBufferPointer { wp in
            n.out[l].withUnsafeBufferPointer { op in
                n.net[l + 1].withUnsafeMutableBufferPointer { netp in
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans,
                        Int32(inSize), Int32(outSize),
                        1.0, wp.baseAddress, Int32(outSize),
                        op.baseAddress, 1,
                        0.0, netp.baseAddress, 1)
                }
            }
        }

        // Activation of layer l+1.
        let id = n.activation[l + 1]
        let scale = n.actScale[l + 1]
        for j in 0..<outSize {
            n.out[l + 1][j] = activate(id, n.net[l + 1][j], scale)
        }
        // Bias slot forced to 1 (not the output layer, which has no bias).
        if !nextIsOutput && n.withBias[l + 1] {
            n.out[l + 1][n.sizes[l + 1] - 1] = 1.0
        }
    }
}

// ============================================================
// C API
// ============================================================

@_cdecl("enn_cpu_available")
public func enn_available() -> Int32 { return 1 }

@_cdecl("enn_cpu_create_network")
public func enn_create_network(
    _ numLayers: Int32,
    _ neuronCounts: UnsafePointer<Int32>,
    _ withBias: UnsafePointer<Int32>,
    _ activationIds: UnsafePointer<Int32>,
    _ activationScale: UnsafePointer<Float>,
    _ flatSpots: UnsafePointer<Float>
) -> UnsafeMutableRawPointer {
    let count = Int(numLayers)
    var sizes = [Int](repeating: 0, count: count)
    var bias = [Bool](repeating: false, count: count)
    var acts = [Int32](repeating: 0, count: count)
    var scales = [Float](repeating: 0, count: count)
    var flats = [Float](repeating: 0, count: count)
    for i in 0..<count {
        sizes[i] = Int(neuronCounts[i])
        bias[i] = withBias[i] != 0
        acts[i] = activationIds[i]
        scales[i] = activationScale[i]
        flats[i] = flatSpots[i]
    }
    let net = EnnNetwork(sizes: sizes, withBias: bias, activation: acts,
                         actScale: scales, flatSpot: flats)
    return Unmanaged.passRetained(net).toOpaque()
}

@_cdecl("enn_cpu_destroy")
public func enn_destroy(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<EnnNetwork>.fromOpaque(ptr).release()
}

@_cdecl("enn_cpu_weights_length")
public func enn_weights_length(_ ptr: UnsafeMutableRawPointer) -> Int32 {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    return Int32(n.weightsLength)
}

// Flat order == ANN.allWeights: hidden layers reversed, then input; each
// connection row-major [inSize][outSize]. In this backend's indexing that is
// layers L = (numLayers-2) down to 0.
@_cdecl("enn_cpu_set_weights")
public func enn_set_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ weights: UnsafePointer<Float>,
    _ length: Int32
) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    var offset = 0
    var l = n.numLayers - 2
    while l >= 0 {
        let count = n.w[l].count
        for i in 0..<count { n.w[l][i] = weights[offset + i] }
        offset += count
        l -= 1
    }
}

@_cdecl("enn_cpu_get_weights")
public func enn_get_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ outWeights: UnsafeMutablePointer<Float>,
    _ length: Int32
) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    var offset = 0
    var l = n.numLayers - 2
    while l >= 0 {
        let count = n.w[l].count
        for i in 0..<count { outWeights[offset + i] = n.w[l][i] }
        offset += count
        l -= 1
    }
}

@_cdecl("enn_cpu_set_samples")
public func enn_set_samples(
    _ ptr: UnsafeMutableRawPointer,
    _ numSamples: Int32,
    _ inputSize: Int32,
    _ outputSize: Int32,
    _ inputs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Float>
) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.numSamples = Int(numSamples)
    n.inputSize = Int(inputSize)
    n.outputSize = Int(outputSize)
    n.inputs = [Float](repeating: 0, count: n.numSamples * n.inputSize)
    n.targets = [Float](repeating: 0, count: n.numSamples * n.outputSize)
    for i in 0..<n.inputs.count { n.inputs[i] = inputs[i] }
    for i in 0..<n.targets.count { n.targets[i] = targets[i] }
}

@_cdecl("enn_cpu_config_backprop")
public func enn_config_backprop(_ ptr: UnsafeMutableRawPointer) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.useRprop = false
}

@_cdecl("enn_cpu_config_rprop")
public func enn_config_rprop(
    _ ptr: UnsafeMutableRawPointer,
    _ delta0: Float, _ deltaMin: Float, _ deltaMax: Float,
    _ etaPlus: Float, _ etaMinus: Float
) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.useRprop = true
    n.rDelta0 = delta0
    n.rMin = deltaMin
    n.rMax = deltaMax
    n.rEtaPlus = etaPlus
    n.rEtaMinus = etaMinus
}

// Resets optimizer state for a fresh training session.
//
// Mirrors `Propagation.reset()` exactly: resets `previousUpdateDeltas` (-> 0.10)
// and `weightsLastUpdates` (-> 0), but does NOT reset the accumulated gradients
// (`g`/`gPrev`) nor the global-learn-error tracking — the pure-Dart `reset()`
// leaves those untouched too.
@_cdecl("enn_cpu_reset_optimizer")
public func enn_reset_optimizer(_ ptr: UnsafeMutableRawPointer) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    for l in 0..<(n.numLayers - 1) {
        let cnt = n.prevDelta[l].count
        for i in 0..<cnt {
            n.prevDelta[l][i] = 0.10
            n.lastUpdate[l][i] = 0
        }
    }
}

// Runs ONE full training epoch and returns the epoch's global learn error.
// `lr`/`momentum` are used only by the Backpropagation update rule.
// If the computed error < target, weights are NOT updated (mirrors `learn`).
@_cdecl("enn_cpu_run_epoch")
public func enn_run_epoch(
    _ ptr: UnsafeMutableRawPointer,
    _ target: Float,
    _ lr: Float,
    _ momentum: Float
) -> Float {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    let last = n.numLayers - 1

    // resetGradients(): roll current -> previous, zero current.
    for l in 0..<(n.numLayers - 1) {
        let cnt = n.g[l].count
        for i in 0..<cnt {
            n.gPrev[l][i] = n.g[l][i]
            n.g[l][i] = 0
        }
    }

    var allSamplesError: Float = 0

    for s in 0..<n.numSamples {
        forwardSample(n, s)

        // Output layer delta + error.
        let outSize = n.sizes[last]
        let outId = n.activation[last]
        let outFlat = n.flatSpot[last]
        for k in 0..<outSize {
            let o = n.out[last][k]
            let err = n.targets[s * n.outputSize + k] - o
            allSamplesError += err * err
            n.delta[last][k] = err * derivative(outId, o, outFlat, true)
        }

        // Middle/input layers (high -> low).
        var l = n.numLayers - 2
        while l >= 0 {
            let inSize = n.sizes[l]
            let outSizeL = n.sizes[l + 1]

            // neuronError = W[l] (NoTrans) * delta[l+1]
            n.w[l].withUnsafeBufferPointer { wp in
                n.delta[l + 1].withUnsafeBufferPointer { dp in
                    n.net[l].withUnsafeMutableBufferPointer { tmp in
                        // reuse net[l] as scratch for neuronError
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(inSize), Int32(outSizeL),
                            1.0, wp.baseAddress, Int32(outSizeL),
                            dp.baseAddress, 1,
                            0.0, tmp.baseAddress, 1)
                    }
                }
            }

            // G[l] += out[l] (outer) delta[l+1]   (rank-1 update)
            n.out[l].withUnsafeBufferPointer { op in
                n.delta[l + 1].withUnsafeBufferPointer { dp in
                    n.g[l].withUnsafeMutableBufferPointer { gp in
                        cblas_sger(
                            CblasRowMajor,
                            Int32(inSize), Int32(outSizeL),
                            1.0,
                            op.baseAddress, 1,
                            dp.baseAddress, 1,
                            gp.baseAddress, Int32(outSizeL))
                    }
                }
            }

            // delta[l][i] = neuronError[i] * deriv(out[l][i])
            let id = n.activation[l]
            let flat = n.flatSpot[l]
            let withFlat = l > 0
            for i in 0..<inSize {
                let d = derivative(id, n.out[l][i], flat, withFlat)
                n.delta[l][i] = n.net[l][i] * d
            }

            l -= 1
        }
    }

    let denom = Float(n.outputSize * n.numSamples)
    let globalError = allSamplesError / denom
    n.lastGlobalLearnError = n.globalLearnError
    n.globalLearnError = globalError

    if globalError < target {
        return globalError
    }

    // Weight update.
    if n.useRprop {
        updateRProp(n)
    } else {
        updateBackprop(n, lr: lr, momentum: momentum)
    }

    return globalError
}

private func updateBackprop(_ n: EnnNetwork, lr: Float, momentum: Float) {
    for l in 0..<(n.numLayers - 1) {
        let cnt = n.w[l].count
        for k in 0..<cnt {
            let delta = lr * n.g[l][k] + momentum * n.prevDelta[l][k]
            n.prevDelta[l][k] = delta
            n.w[l][k] += delta
        }
    }
}

private func updateRProp(_ n: EnnNetwork) {
    let tol: Float = 1.0e-20
    let backtrack = n.globalLearnError > n.lastGlobalLearnError
    for l in 0..<(n.numLayers - 1) {
        let cnt = n.w[l].count
        for k in 0..<cnt {
            let grad = n.g[l][k]
            let prevGrad = n.gPrev[l][k]
            var pd = n.prevDelta[l][k]
            var change = signZeroTolerance(grad * prevGrad, tol)
            let gradSign = signZeroTolerance(grad, tol)

            if pd < 0 {
                pd = -pd
                change = 0
            }

            var updateDelta: Float
            var weightUpdate: Float
            if change > 0 {
                updateDelta = min(pd * n.rEtaPlus, n.rMax)
                weightUpdate = gradSign * updateDelta
            } else if change < 0 {
                updateDelta = max(pd * n.rEtaMinus, n.rMin)
                updateDelta = -updateDelta
                weightUpdate = backtrack ? (n.lastUpdate[l][k] * -1.0) : 0.0
            } else {
                updateDelta = pd
                weightUpdate = gradSign * updateDelta
            }

            n.prevDelta[l][k] = updateDelta
            n.lastUpdate[l][k] = weightUpdate
            n.w[l][k] += weightUpdate
        }
    }
}

// Inference: forward `input` -> `output`.
@_cdecl("enn_cpu_activate")
public func enn_activate(
    _ ptr: UnsafeMutableRawPointer,
    _ input: UnsafePointer<Float>,
    _ inputSize: Int32,
    _ output: UnsafeMutablePointer<Float>,
    _ outputSize: Int32
) {
    let n = Unmanaged<EnnNetwork>.fromOpaque(ptr).takeUnretainedValue()
    let inSize0 = Int(inputSize)
    for i in 0..<inSize0 { n.out[0][i] = input[i] }
    if n.withBias[0] { n.out[0][n.sizes[0] - 1] = 1.0 }

    for l in 0..<(n.numLayers - 1) {
        let inSize = n.sizes[l]
        let outSize = n.sizes[l + 1]
        let nextIsOutput = (l + 1) == (n.numLayers - 1)
        n.w[l].withUnsafeBufferPointer { wp in
            n.out[l].withUnsafeBufferPointer { op in
                n.net[l + 1].withUnsafeMutableBufferPointer { netp in
                    cblas_sgemv(
                        CblasRowMajor, CblasTrans,
                        Int32(inSize), Int32(outSize),
                        1.0, wp.baseAddress, Int32(outSize),
                        op.baseAddress, 1,
                        0.0, netp.baseAddress, 1)
                }
            }
        }
        let id = n.activation[l + 1]
        let scale = n.actScale[l + 1]
        for j in 0..<outSize {
            n.out[l + 1][j] = activate(id, n.net[l + 1][j], scale)
        }
        if !nextIsOutput && n.withBias[l + 1] {
            n.out[l + 1][n.sizes[l + 1] - 1] = 1.0
        }
    }

    let last = n.numLayers - 1
    let outCount = Int(outputSize)
    for k in 0..<outCount { output[k] = n.out[last][k] }
}
