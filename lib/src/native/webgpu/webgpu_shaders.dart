// WGSL compute shaders for the WebGPU whole-epoch trainer.
//
// The trainer is batched: all samples are processed at once (buffers are laid
// out row-major, sample-major: index = sample * layerSize + feature). Each
// kernel is a standalone module with entry point `main` and its own bindings
// (storage buffers at bindings 0..k-1, a shared 64-byte uniform at binding k).
//
// The kernels reproduce the pure-Dart numerics exactly (activation/derivative
// formulas, the bias-row=1 rule, Backpropagation and iRProp+ updates).

/// Common WGSL prelude: the uniform params struct and helper functions.
const String _wgslCommon = '''
struct Params {
  i0: i32, i1: i32, i2: i32, i3: i32,
  i4: i32, i5: i32, i6: i32, i7: i32,
  f0: f32, f1: f32, f2: f32, f3: f32,
  f4: f32, f5: f32, f6: f32, f7: f32,
};

fn ennActivate(id: i32, x: f32, scale: f32) -> f32 {
  if (id == 0) { return x; }
  if (id == 1) { return 1.0 / (1.0 + exp(-x)); }
  if (id == 2) { let x3 = x * 3.0; return 0.5 + (x3 / (2.5 + abs(x3)) / 2.0); }
  let v = clamp(x, -scale, scale) / scale;
  return 0.5 + (v / (1.0 + v * v));
}

fn ennDeriv(id: i32, o: f32, flat: f32, withFlat: bool) -> f32 {
  if (id == 0) {
    if (withFlat) { return 1.0 + flat; }
    return 1.0;
  }
  let d = o * (1.0 - o);
  if (withFlat) { return d + flat; }
  return d;
}

fn ennSignZt(v: f32, tol: f32) -> f32 {
  if (v > 0.0) {
    if (v < tol) { return 0.0; }
    return 1.0;
  }
  if (v > -tol) { return 0.0; }
  return -1.0;
}
''';

/// zero(buf): i0 = n
const String wgslZero =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> P: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i < P.i0) { buf[i] = 0.0; }
}
''';

/// copy(dst, src): i0 = n
const String wgslCopy =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i < P.i0) { dst[i] = src[i]; }
}
''';

/// loadInput(out0, inputs): i0=layerSize, i1=inputSize, i2=biasIndex, i3=numSamples
const String wgslLoadInput =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> out0: array<f32>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = i32(gid.x);
  let s = i32(gid.y);
  if (c >= P.i0 || s >= P.i3) { return; }
  var v: f32;
  if (c == P.i2) { v = 1.0; }
  else if (c < P.i1) { v = inputs[s * P.i1 + c]; }
  else { v = 0.0; }
  out0[s * P.i0 + c] = v;
}
''';

/// forward(w, outPrev, outNext): matmul + activate + bias.
/// i0=inSize, i1=outSize, i2=actId, i3=biasIndex, i4=numSamples, f0=scale
const String wgslForward =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> outPrev: array<f32>;
@group(0) @binding(2) var<storage, read_write> outNext: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = i32(gid.x);
  let s = i32(gid.y);
  if (j >= P.i1 || s >= P.i4) { return; }
  var acc = 0.0;
  for (var i = 0; i < P.i0; i = i + 1) {
    acc = acc + w[i * P.i1 + j] * outPrev[s * P.i0 + i];
  }
  var o: f32;
  if (j == P.i3) { o = 1.0; }
  else { o = ennActivate(P.i2, acc, P.f0); }
  outNext[s * P.i1 + j] = o;
}
''';

/// outputDelta(outLast, targets, delta, errBuf): i0=N, i1=actId, i2=numSamples, f0=flat
const String wgslOutputDelta =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read> outLast: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> delta: array<f32>;
@group(0) @binding(3) var<storage, read_write> errBuf: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = i32(gid.x);
  let s = i32(gid.y);
  if (k >= P.i0 || s >= P.i2) { return; }
  let idx = s * P.i0 + k;
  let o = outLast[idx];
  let err = targets[idx] - o;
  errBuf[idx] = err * err;
  delta[idx] = err * ennDeriv(P.i1, o, P.f0, true);
}
''';

/// backprop(w, deltaNext, outCur, deltaCur): matmul + derivative.
/// i0=inSize, i1=outSize, i2=actId, i3=withFlat, i4=numSamples, f0=flat
const String wgslBackprop =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> deltaNext: array<f32>;
@group(0) @binding(2) var<storage, read> outCur: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltaCur: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let s = i32(gid.y);
  if (i >= P.i0 || s >= P.i4) { return; }
  var acc = 0.0;
  for (var j = 0; j < P.i1; j = j + 1) {
    acc = acc + w[i * P.i1 + j] * deltaNext[s * P.i1 + j];
  }
  deltaCur[s * P.i0 + i] = acc * ennDeriv(P.i2, outCur[s * P.i0 + i], P.f0, P.i3 != 0);
}
''';

/// gradient(g, outCur, deltaNext): batch-summed. i0=inSize, i1=outSize, i2=numSamples
const String wgslGradient =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> outCur: array<f32>;
@group(0) @binding(2) var<storage, read> deltaNext: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = i32(gid.x);
  let i = i32(gid.y);
  if (j >= P.i1 || i >= P.i0) { return; }
  var acc = 0.0;
  for (var s = 0; s < P.i2; s = s + 1) {
    acc = acc + outCur[s * P.i0 + i] * deltaNext[s * P.i1 + j];
  }
  g[i * P.i1 + j] = acc;
}
''';

/// updateBp(w, g, prevDelta): i0=n, f0=lr, f1=momentum
const String wgslUpdateBp =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> prevDelta: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i >= P.i0) { return; }
  let d = P.f0 * g[i] + P.f1 * prevDelta[i];
  prevDelta[i] = d;
  w[i] = w[i] + d;
}
''';

/// updateRprop(w, g, gPrev, prevDelta, lastUpdate):
/// i0=backtrack, i1=n, f0=etaPlus, f1=etaMinus, f2=dMin, f3=dMax
const String wgslUpdateRprop =
    '''
$_wgslCommon
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> gPrev: array<f32>;
@group(0) @binding(3) var<storage, read_write> prevDelta: array<f32>;
@group(0) @binding(4) var<storage, read_write> lastUpdate: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i >= P.i1) { return; }
  let tol = 1e-20;
  let grad = g[i];
  let pg = gPrev[i];
  var pd = prevDelta[i];
  var change = ennSignZt(grad * pg, tol);
  let gs = ennSignZt(grad, tol);
  if (pd < 0.0) { pd = -pd; change = 0.0; }
  var ud: f32;
  var wu: f32;
  if (change > 0.0) {
    ud = min(pd * P.f0, P.f3);
    wu = gs * ud;
  } else if (change < 0.0) {
    ud = max(pd * P.f1, P.f2);
    ud = -ud;
    if (P.i0 != 0) { wu = lastUpdate[i] * -1.0; }
    else { wu = 0.0; }
  } else {
    ud = pd;
    wu = gs * ud;
  }
  prevDelta[i] = ud;
  lastUpdate[i] = wu;
  w[i] = w[i] + wu;
}
''';
