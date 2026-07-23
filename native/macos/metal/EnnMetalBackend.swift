// eneural_net — native Metal (GPU) backend, BATCHED whole-epoch trainer.
//
// All samples are processed at once as matrices. Each epoch is:
//   forward:   NET_{L+1} = OUT_L · W_L
//   backprop:  DELTA_L   = DELTA_{L+1} · W_L^T
//   gradient:  G_L       = OUT_L^T · DELTA_{L+1}   (batch-summed in one GEMM)
// The GEMMs use MetalPerformanceShaders; activation/delta/update are elementwise
// kernels (EnnMetalShaders). Dispatch count is O(layers), independent of the
// sample count — the previous per-sample design was latency-bound.
//
// The kernels reproduce the pure-Dart numerics (incl. the bias-row=1 rule) so
// weights read back match the Dart trainer within float32 tolerance.
//
// Activation ids: 0=Linear, 1=Sigmoid, 2=SigmoidFast, 3=SigmoidBoundedFast.
// C symbols are prefixed `enn_metal_` so this dylib can coexist with the CPU one.

import Foundation
import Metal
import MetalPerformanceShaders

// ============================================================
// Context: device / queue / runtime-compiled library / pipeline cache
// ============================================================

final class EnnMetalContext {
    static let shared: EnnMetalContext? = EnnMetalContext()

    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            return nil
        }
        do {
            self.library = try device.makeLibrary(
                source: EnnMetalShaders.source, options: nil)
        } catch {
            return nil
        }
        self.device = device
        self.queue = queue
    }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        let fn = library.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }
}

// ============================================================
// Resident network state
// ============================================================

final class EnnMetalNetwork {
    let ctx: EnnMetalContext
    let numLayers: Int
    let sizes: [Int]
    let withBias: [Bool]
    let activation: [Int32]
    let actScale: [Float]
    let flatSpot: [Float]

    // Per connection L -> L+1 (row-major [inSize][outSize]).
    var w: [MTLBuffer] = []
    var g: [MTLBuffer] = []
    var gPrev: [MTLBuffer] = []
    var prevDelta: [MTLBuffer] = []
    var lastUpdate: [MTLBuffer] = []
    var wCounts: [Int] = []

    // Single-sample inference buffers.
    var outInf: [MTLBuffer] = []

    // Batched activation buffers (per layer, [numSamples x sizes[L]]).
    var out: [MTLBuffer] = []
    var net: [MTLBuffer] = []
    var delta: [MTLBuffer] = []

    // Samples (resident).
    var numSamples = 0
    var inputSize = 0
    var outputSize = 0
    var inputs: MTLBuffer?
    var targets: MTLBuffer?
    var errBuf: MTLBuffer?

    // MPS matrices/multiplications (built once samples are known).
    var mpsReady = false
    var matOut: [MPSMatrix] = []
    var matNet: [MPSMatrix] = []
    var matDelta: [MPSMatrix] = []
    var matW: [MPSMatrix] = []
    var matG: [MPSMatrix] = []
    var fwdMul: [MPSMatrixMultiplication] = []
    var bpMul: [MPSMatrixMultiplication] = []
    var gradMul: [MPSMatrixMultiplication] = []

    // Update rule.
    var useRprop = false
    var rDelta0: Float = 0.10
    var rMin: Float = 1.0e-6
    var rMax: Float = 50.0
    var rEtaPlus: Float = 1.2
    var rEtaMinus: Float = 0.5

    var globalLearnError: Float = 1.0
    var lastGlobalLearnError: Float = 1.0

    init(ctx: EnnMetalContext, sizes: [Int], withBias: [Bool],
         activation: [Int32], actScale: [Float], flatSpot: [Float]) {
        self.ctx = ctx
        self.numLayers = sizes.count
        self.sizes = sizes
        self.withBias = withBias
        self.activation = activation
        self.actScale = actScale
        self.flatSpot = flatSpot

        let dev = ctx.device
        func buf(_ count: Int, fill: Float = 0) -> MTLBuffer {
            let b = dev.makeBuffer(length: max(count, 1) * 4,
                                   options: .storageModeShared)!
            let p = b.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { p[i] = fill }
            return b
        }

        for l in 0..<(numLayers - 1) {
            let n = sizes[l] * sizes[l + 1]
            wCounts.append(n)
            w.append(buf(n))
            g.append(buf(n))
            gPrev.append(buf(n))
            prevDelta.append(buf(n, fill: 0.10))
            lastUpdate.append(buf(n))
        }
        for l in 0..<numLayers {
            outInf.append(buf(sizes[l]))
        }
    }

    var weightsLength: Int { wCounts.reduce(0, +) }

    private func fbuf(_ count: Int) -> MTLBuffer {
        ctx.device.makeBuffer(length: max(count, 1) * 4,
                              options: .storageModeShared)!
    }

    func allocateBatchBuffers() {
        out = sizes.map { fbuf(numSamples * $0) }
        net = sizes.map { fbuf(numSamples * $0) }
        delta = sizes.map { fbuf(numSamples * $0) }
    }

    func buildMPS() {
        if mpsReady { return }
        let m = numSamples

        func desc(_ rows: Int, _ cols: Int) -> MPSMatrixDescriptor {
            MPSMatrixDescriptor(rows: rows, columns: cols,
                                rowBytes: cols * 4, dataType: .float32)
        }

        matOut = (0..<numLayers).map {
            MPSMatrix(buffer: out[$0], descriptor: desc(m, sizes[$0]))
        }
        matNet = (0..<numLayers).map {
            MPSMatrix(buffer: net[$0], descriptor: desc(m, sizes[$0]))
        }
        matDelta = (0..<numLayers).map {
            MPSMatrix(buffer: delta[$0], descriptor: desc(m, sizes[$0]))
        }
        matW = (0..<(numLayers - 1)).map {
            MPSMatrix(buffer: w[$0], descriptor: desc(sizes[$0], sizes[$0 + 1]))
        }
        matG = (0..<(numLayers - 1)).map {
            MPSMatrix(buffer: g[$0], descriptor: desc(sizes[$0], sizes[$0 + 1]))
        }

        let dev = ctx.device
        fwdMul = (0..<(numLayers - 1)).map { l in
            MPSMatrixMultiplication(
                device: dev, transposeLeft: false, transposeRight: false,
                resultRows: m, resultColumns: sizes[l + 1],
                interiorColumns: sizes[l], alpha: 1.0, beta: 0.0)
        }
        bpMul = (0..<(numLayers - 1)).map { l in
            MPSMatrixMultiplication(
                device: dev, transposeLeft: false, transposeRight: true,
                resultRows: m, resultColumns: sizes[l],
                interiorColumns: sizes[l + 1], alpha: 1.0, beta: 0.0)
        }
        gradMul = (0..<(numLayers - 1)).map { l in
            MPSMatrixMultiplication(
                device: dev, transposeLeft: true, transposeRight: false,
                resultRows: sizes[l], resultColumns: sizes[l + 1],
                interiorColumns: m, alpha: 1.0, beta: 0.0)
        }

        mpsReady = true
    }
}

// ============================================================
// Dispatch helpers
// ============================================================

private func runK1D(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                    _ count: Int, _ bind: (MTLComputeCommandEncoder) -> Void) {
    if count <= 0 { return }
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    bind(enc)
    let w = min(pipe.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(w, count),
                                                       height: 1, depth: 1))
    enc.endEncoding()
}

private func runK2D(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                    _ gridW: Int, _ gridH: Int,
                    _ bind: (MTLComputeCommandEncoder) -> Void) {
    if gridW <= 0 || gridH <= 0 { return }
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    bind(enc)
    let tw = min(pipe.threadExecutionWidth, gridW)
    let th = max(1, min(pipe.maxTotalThreadsPerThreadgroup / max(tw, 1), gridH))
    enc.dispatchThreads(
        MTLSize(width: gridW, height: gridH, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    enc.endEncoding()
}

private func setInt(_ enc: MTLComputeCommandEncoder, _ v: Int, _ index: Int) {
    var x = Int32(v)
    enc.setBytes(&x, length: 4, index: index)
}

private func setFloat(_ enc: MTLComputeCommandEncoder, _ v: Float, _ index: Int) {
    var x = v
    enc.setBytes(&x, length: 4, index: index)
}

// ============================================================
// C API
// ============================================================

@_cdecl("enn_metal_available")
public func enn_metal_available() -> Int32 {
    return EnnMetalContext.shared != nil ? 1 : 0
}

@_cdecl("enn_metal_create_network")
public func enn_metal_create_network(
    _ numLayers: Int32,
    _ neuronCounts: UnsafePointer<Int32>,
    _ withBias: UnsafePointer<Int32>,
    _ activationIds: UnsafePointer<Int32>,
    _ activationScale: UnsafePointer<Float>,
    _ flatSpots: UnsafePointer<Float>
) -> UnsafeMutableRawPointer? {
    guard let ctx = EnnMetalContext.shared else { return nil }
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
    let net = EnnMetalNetwork(ctx: ctx, sizes: sizes, withBias: bias,
                              activation: acts, actScale: scales, flatSpot: flats)
    return Unmanaged.passRetained(net).toOpaque()
}

@_cdecl("enn_metal_destroy")
public func enn_metal_destroy(_ ptr: UnsafeMutableRawPointer) {
    Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).release()
}

@_cdecl("enn_metal_weights_length")
public func enn_metal_weights_length(_ ptr: UnsafeMutableRawPointer) -> Int32 {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    return Int32(n.weightsLength)
}

// Flat order == ANN.allWeights: layers L = (numLayers-2) down to 0.
@_cdecl("enn_metal_set_weights")
public func enn_metal_set_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ weights: UnsafePointer<Float>,
    _ length: Int32
) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    var offset = 0
    var l = n.numLayers - 2
    while l >= 0 {
        let count = n.wCounts[l]
        let dst = n.w[l].contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { dst[i] = weights[offset + i] }
        offset += count
        l -= 1
    }
}

@_cdecl("enn_metal_get_weights")
public func enn_metal_get_weights(
    _ ptr: UnsafeMutableRawPointer,
    _ outWeights: UnsafeMutablePointer<Float>,
    _ length: Int32
) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    var offset = 0
    var l = n.numLayers - 2
    while l >= 0 {
        let count = n.wCounts[l]
        let src = n.w[l].contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { outWeights[offset + i] = src[i] }
        offset += count
        l -= 1
    }
}

@_cdecl("enn_metal_set_samples")
public func enn_metal_set_samples(
    _ ptr: UnsafeMutableRawPointer,
    _ numSamples: Int32,
    _ inputSize: Int32,
    _ outputSize: Int32,
    _ inputs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Float>
) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.numSamples = Int(numSamples)
    n.inputSize = Int(inputSize)
    n.outputSize = Int(outputSize)
    let inCount = n.numSamples * n.inputSize
    let outCount = n.numSamples * n.outputSize
    let dev = n.ctx.device

    let inBuf = dev.makeBuffer(length: max(inCount, 1) * 4,
                               options: .storageModeShared)!
    let tBuf = dev.makeBuffer(length: max(outCount, 1) * 4,
                              options: .storageModeShared)!
    let eBuf = dev.makeBuffer(length: max(outCount, 1) * 4,
                              options: .storageModeShared)!
    let ip = inBuf.contents().assumingMemoryBound(to: Float.self)
    let tp = tBuf.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<inCount { ip[i] = inputs[i] }
    for i in 0..<outCount { tp[i] = targets[i] }
    n.inputs = inBuf
    n.targets = tBuf
    n.errBuf = eBuf

    n.allocateBatchBuffers()
    n.mpsReady = false
    n.buildMPS()
}

@_cdecl("enn_metal_config_backprop")
public func enn_metal_config_backprop(_ ptr: UnsafeMutableRawPointer) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.useRprop = false
}

@_cdecl("enn_metal_config_rprop")
public func enn_metal_config_rprop(
    _ ptr: UnsafeMutableRawPointer,
    _ delta0: Float, _ deltaMin: Float, _ deltaMax: Float,
    _ etaPlus: Float, _ etaMinus: Float
) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    n.useRprop = true
    n.rDelta0 = delta0
    n.rMin = deltaMin
    n.rMax = deltaMax
    n.rEtaPlus = etaPlus
    n.rEtaMinus = etaMinus
}

// Mirrors Propagation.reset(): prevDelta -> 0.10, lastUpdate -> 0.
@_cdecl("enn_metal_reset_optimizer")
public func enn_metal_reset_optimizer(_ ptr: UnsafeMutableRawPointer) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    for l in 0..<(n.numLayers - 1) {
        let count = n.wCounts[l]
        let pd = n.prevDelta[l].contents().assumingMemoryBound(to: Float.self)
        let lu = n.lastUpdate[l].contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count { pd[i] = 0.10; lu[i] = 0.0 }
    }
}

@_cdecl("enn_metal_run_epoch")
public func enn_metal_run_epoch(
    _ ptr: UnsafeMutableRawPointer,
    _ target: Float,
    _ lr: Float,
    _ momentum: Float
) -> Float {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    let ctx = n.ctx
    let last = n.numLayers - 1
    guard let inputs = n.inputs, let errBuf = n.errBuf, n.mpsReady else {
        return n.globalLearnError
    }

    let zeroPipe = ctx.pipeline("enn_k_zero")
    let copyPipe = ctx.pipeline("enn_k_copy")
    let loadPipe = ctx.pipeline("enn_k_load_input_batch")
    let actPipe = ctx.pipeline("enn_k_activate_batch")
    let outDeltaPipe = ctx.pipeline("enn_k_output_delta_batch")
    let deltaPipe = ctx.pipeline("enn_k_delta_batch")

    let m = n.numSamples
    let errCount = n.numSamples * n.outputSize

    // ---- Pass 1: forward + backprop + gradient (all samples batched).
    let cmd1 = ctx.queue.makeCommandBuffer()!

    runK1D(cmd1, zeroPipe, errCount) { enc in
        enc.setBuffer(errBuf, offset: 0, index: 0)
        setInt(enc, errCount, 1)
    }

    // resetGradients(): gPrev = g (current becomes previous; new g is overwritten
    // by the gradient GEMM below).
    for l in 0..<(n.numLayers - 1) {
        runK1D(cmd1, copyPipe, n.wCounts[l]) { enc in
            enc.setBuffer(n.gPrev[l], offset: 0, index: 0)
            enc.setBuffer(n.g[l], offset: 0, index: 1)
            setInt(enc, n.wCounts[l], 2)
        }
    }

    // Load input batch into out[0].
    let biasIndex0 = n.withBias[0] ? n.sizes[0] - 1 : -1
    runK2D(cmd1, loadPipe, n.sizes[0], m) { enc in
        enc.setBuffer(n.out[0], offset: 0, index: 0)
        enc.setBuffer(inputs, offset: 0, index: 1)
        setInt(enc, n.sizes[0], 2)
        setInt(enc, n.inputSize, 3)
        setInt(enc, biasIndex0, 4)
        setInt(enc, m, 5)
    }

    // Forward.
    for l in 0..<(n.numLayers - 1) {
        n.fwdMul[l].encode(commandBuffer: cmd1, leftMatrix: n.matOut[l],
                           rightMatrix: n.matW[l], resultMatrix: n.matNet[l + 1])
        let nextIsOutput = (l + 1) == last
        let biasIndex =
            (!nextIsOutput && n.withBias[l + 1]) ? (n.sizes[l + 1] - 1) : -1
        runK2D(cmd1, actPipe, n.sizes[l + 1], m) { enc in
            enc.setBuffer(n.net[l + 1], offset: 0, index: 0)
            enc.setBuffer(n.out[l + 1], offset: 0, index: 1)
            setInt(enc, n.sizes[l + 1], 2)
            setInt(enc, Int(n.activation[l + 1]), 3)
            setFloat(enc, n.actScale[l + 1], 4)
            setInt(enc, biasIndex, 5)
            setInt(enc, m, 6)
        }
    }

    // Output delta + error.
    runK2D(cmd1, outDeltaPipe, n.sizes[last], m) { enc in
        enc.setBuffer(n.out[last], offset: 0, index: 0)
        enc.setBuffer(n.targets!, offset: 0, index: 1)
        enc.setBuffer(n.delta[last], offset: 0, index: 2)
        enc.setBuffer(errBuf, offset: 0, index: 3)
        setInt(enc, n.sizes[last], 4)
        setInt(enc, Int(n.activation[last]), 5)
        setFloat(enc, n.flatSpot[last], 6)
        setInt(enc, m, 7)
    }

    // Backprop (high -> low): gradient for every layer, delta for hidden layers.
    var l = n.numLayers - 2
    while l >= 0 {
        // G[l] = OUT[l]^T · DELTA[l+1]  (batch-summed gradient).
        n.gradMul[l].encode(commandBuffer: cmd1, leftMatrix: n.matOut[l],
                            rightMatrix: n.matDelta[l + 1], resultMatrix: n.matG[l])

        if l > 0 {
            // neuronError DELTA[l] = DELTA[l+1] · W[l]^T
            n.bpMul[l].encode(commandBuffer: cmd1, leftMatrix: n.matDelta[l + 1],
                             rightMatrix: n.matW[l], resultMatrix: n.matDelta[l])
            runK2D(cmd1, deltaPipe, n.sizes[l], m) { enc in
                enc.setBuffer(n.delta[l], offset: 0, index: 0)
                enc.setBuffer(n.out[l], offset: 0, index: 1)
                setInt(enc, n.sizes[l], 2)
                setInt(enc, Int(n.activation[l]), 3)
                setFloat(enc, n.flatSpot[l], 4)
                setInt(enc, 1, 5) // l > 0 -> withFlatSpot
                setInt(enc, m, 6)
            }
        }
        l -= 1
    }

    cmd1.commit()
    cmd1.waitUntilCompleted()

    // Global error.
    let ep = errBuf.contents().assumingMemoryBound(to: Float.self)
    var sumSq: Float = 0
    for i in 0..<errCount { sumSq += ep[i] }
    let globalError = sumSq / Float(n.outputSize * n.numSamples)
    n.lastGlobalLearnError = n.globalLearnError
    n.globalLearnError = globalError

    if globalError < target {
        return globalError
    }

    // ---- Pass 2: weight update.
    let cmd2 = ctx.queue.makeCommandBuffer()!
    if n.useRprop {
        let pipe = ctx.pipeline("enn_k_update_rprop")
        let backtrack = n.globalLearnError > n.lastGlobalLearnError ? 1 : 0
        for li in 0..<(n.numLayers - 1) {
            runK1D(cmd2, pipe, n.wCounts[li]) { enc in
                enc.setBuffer(n.w[li], offset: 0, index: 0)
                enc.setBuffer(n.g[li], offset: 0, index: 1)
                enc.setBuffer(n.gPrev[li], offset: 0, index: 2)
                enc.setBuffer(n.prevDelta[li], offset: 0, index: 3)
                enc.setBuffer(n.lastUpdate[li], offset: 0, index: 4)
                setFloat(enc, n.rEtaPlus, 5)
                setFloat(enc, n.rEtaMinus, 6)
                setFloat(enc, n.rMin, 7)
                setFloat(enc, n.rMax, 8)
                setInt(enc, backtrack, 9)
                setInt(enc, n.wCounts[li], 10)
            }
        }
    } else {
        let pipe = ctx.pipeline("enn_k_update_bp")
        for li in 0..<(n.numLayers - 1) {
            runK1D(cmd2, pipe, n.wCounts[li]) { enc in
                enc.setBuffer(n.w[li], offset: 0, index: 0)
                enc.setBuffer(n.g[li], offset: 0, index: 1)
                enc.setBuffer(n.prevDelta[li], offset: 0, index: 2)
                setFloat(enc, lr, 3)
                setFloat(enc, momentum, 4)
                setInt(enc, n.wCounts[li], 5)
            }
        }
    }
    cmd2.commit()
    cmd2.waitUntilCompleted()

    return globalError
}

@_cdecl("enn_metal_activate")
public func enn_metal_activate(
    _ ptr: UnsafeMutableRawPointer,
    _ input: UnsafePointer<Float>,
    _ inputSize: Int32,
    _ output: UnsafeMutablePointer<Float>,
    _ outputSize: Int32
) {
    let n = Unmanaged<EnnMetalNetwork>.fromOpaque(ptr).takeUnretainedValue()
    let ctx = n.ctx
    let last = n.numLayers - 1

    let out0 = n.outInf[0].contents().assumingMemoryBound(to: Float.self)
    let inCount = Int(inputSize)
    for i in 0..<inCount { out0[i] = input[i] }
    if n.withBias[0] { out0[n.sizes[0] - 1] = 1.0 }

    let cmd = ctx.queue.makeCommandBuffer()!
    let fwdPipe = ctx.pipeline("enn_k_forward")
    for l in 0..<(n.numLayers - 1) {
        let inSize = n.sizes[l]
        let outSize = n.sizes[l + 1]
        let nextIsOutput = (l + 1) == last
        let biasIndex =
            (!nextIsOutput && n.withBias[l + 1]) ? (n.sizes[l + 1] - 1) : -1
        runK1D(cmd, fwdPipe, outSize) { enc in
            enc.setBuffer(n.w[l], offset: 0, index: 0)
            enc.setBuffer(n.outInf[l], offset: 0, index: 1)
            enc.setBuffer(n.outInf[l + 1], offset: 0, index: 2)
            setInt(enc, inSize, 3)
            setInt(enc, outSize, 4)
            setInt(enc, Int(n.activation[l + 1]), 5)
            setFloat(enc, n.actScale[l + 1], 6)
            setInt(enc, biasIndex, 7)
        }
    }
    cmd.commit()
    cmd.waitUntilCompleted()

    let outp = n.outInf[last].contents().assumingMemoryBound(to: Float.self)
    let outCount = Int(outputSize)
    for k in 0..<outCount { output[k] = outp[k] }
}
