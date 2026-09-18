// gpu-paste compute kernels. Mirrors src/cpu.ts / training/gpu_paste/model.py exactly.
//
// Passes (dispatched in order, one bind group shared by all of them):
//   embed       x_t = sum_f E[id_f]                         one thread per (token, channel)
//   gates       a, (1-a)*u for both scan directions          one thread per (dir, token, channel)
//   scan_local  Hillis-Steele affine prefix scan inside      one workgroup per (chunk, dir, channel)
//               256-token chunks (workgroup memory)
//   scan_fixup  compose chunk carries, write h               one thread per (dir, token, channel)
//   mix         m_t = relu(W_mix [x; hf; hb] + b)            one workgroup per token
//   pool        g = mean_t m_t, kind logits                  one workgroup
//   head        span logits from [m_t; g]                    one workgroup per token
//
// Every entry point statically references every binding because the runtime creates one
// bind group from the first pipeline's auto layout and reuses it for all passes.

const DIM: u32 = 32u;      // embedding / scan channels
const MIX: u32 = 48u;      // mixed hidden size
const KH: u32 = 32u;       // kind-head hidden size
const CHUNK: u32 = 256u;   // scan chunk length (= workgroup size)

struct Params {
  tokens: u32,
  chunks: u32,
  labels: u32,
  kinds: u32,
  features: u32,
  off_embed: u32,
  off_gate_f_w: u32,
  off_gate_f_b: u32,
  off_gate_b_w: u32,
  off_gate_b_b: u32,
  off_mix_w: u32,
  off_mix_b: u32,
  off_head_w: u32,
  off_head_b: u32,
  off_out_w: u32,
  off_out_b: u32,
  off_kind1_w: u32,
  off_kind1_b: u32,
  off_kind2_w: u32,
  off_kind2_b: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
// state: x [tokens*DIM] | m [tokens*MIX] | g [MIX] | kind hidden [KH]
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
// scan: coefA [2*tokens*DIM] | coefB [2*tokens*DIM] | h [2*tokens*DIM]; index dir*tokens*DIM + t*DIM + d
@group(0) @binding(4) var<storage, read_write> scan: array<f32>;
// logits: span [tokens*labels] | kind [kinds]
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;

var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_s: array<f32, 64>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn relu(x: f32) -> f32 { return max(x, 0.0); }

fn m_off() -> u32 { return params.tokens * DIM; }
fn g_off() -> u32 { return params.tokens * (DIM + MIX); }
fn kh_off() -> u32 { return g_off() + MIX; }
fn coef_a(dir: u32, t: u32, d: u32) -> u32 { return dir * params.tokens * DIM + t * DIM + d; }
fn coef_b(dir: u32, t: u32, d: u32) -> u32 { return 2u * params.tokens * DIM + coef_a(dir, t, d); }
fn hidden(dir: u32, t: u32, d: u32) -> u32 { return 4u * params.tokens * DIM + coef_a(dir, t, d); }

// Static reference to every binding (see header comment); folded away by the compiler.
fn touch() -> f32 {
  return f32(params.tokens) + f32(features[0]) + weights[0] + state[0] + scan[0] + logits[0];
}

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let n = params.tokens;
  if (id.x >= n * DIM) { return; }
  let t = id.x / DIM;
  let d = id.x % DIM;
  var acc = 0.0;
  for (var f = 0u; f < params.features; f = f + 1u) {
    let row = features[t * params.features + f];
    acc = acc + weights[params.off_embed + row * DIM + d];
  }
  state[t * DIM + d] = acc;
}

@compute @workgroup_size(64)
fn gates(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let n = params.tokens;
  if (id.x >= 2u * n * DIM) { return; }
  let dir = id.x / (n * DIM);
  let rem = id.x % (n * DIM);
  let t = rem / DIM;
  let d = rem % DIM;
  var w_off = params.off_gate_f_w;
  var b_off = params.off_gate_f_b;
  if (dir == 1u) { w_off = params.off_gate_b_w; b_off = params.off_gate_b_b; }
  var pa = weights[b_off + d];
  var pu = weights[b_off + DIM + d];
  for (var i = 0u; i < DIM; i = i + 1u) {
    let xi = state[t * DIM + i];
    pa = pa + weights[w_off + d * DIM + i] * xi;
    pu = pu + weights[w_off + (DIM + d) * DIM + i] * xi;
  }
  let a = sigmoid(pa);
  scan[coef_a(dir, t, d)] = a;
  scan[coef_b(dir, t, d)] = (1.0 - a) * tanh(pu);
}

// Position `p` in direction order maps to token index t (forward: p, backward: n-1-p).
fn token_at(dir: u32, p: u32) -> u32 {
  if (dir == 0u) { return p; }
  return params.tokens - 1u - p;
}

@compute @workgroup_size(256)
fn scan_local(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let n = params.tokens;
  let chunk = wg.x;
  let dir = wg.y / DIM;
  let d = wg.y % DIM;
  let p = lid.x;
  let pos = chunk * CHUNK + p;
  let valid = pos < n;
  var a = 1.0;
  var b = 0.0;
  if (valid) {
    let t = token_at(dir, pos);
    a = scan[coef_a(dir, t, d)];
    b = scan[coef_b(dir, t, d)];
  }
  wg_a[p] = a;
  wg_b[p] = b;
  workgroupBarrier();
  for (var k = 1u; k < CHUNK; k = k << 1u) {
    var na = wg_a[p];
    var nb = wg_b[p];
    if (p >= k) {
      // compose earlier map (p-k) then this map (p): h -> a2*(a1*h + b1) + b2
      let a1 = wg_a[p - k];
      let b1 = wg_b[p - k];
      na = a1 * wg_a[p];
      nb = wg_a[p] * b1 + wg_b[p];
    }
    workgroupBarrier();
    wg_a[p] = na;
    wg_b[p] = nb;
    workgroupBarrier();
  }
  if (valid) {
    let t = token_at(dir, pos);
    scan[coef_a(dir, t, d)] = wg_a[p];
    scan[coef_b(dir, t, d)] = wg_b[p];
  }
}

@compute @workgroup_size(64)
fn scan_fixup(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let n = params.tokens;
  if (id.x >= 2u * n * DIM) { return; }
  let dir = id.x / (n * DIM);
  let rem = id.x % (n * DIM);
  let pos = rem / DIM;
  let d = rem % DIM;
  let chunk = pos / CHUNK;
  var carry = 0.0;
  for (var c = 0u; c < chunk; c = c + 1u) {
    let last = token_at(dir, c * CHUNK + CHUNK - 1u);
    carry = scan[coef_a(dir, last, d)] * carry + scan[coef_b(dir, last, d)];
  }
  let t = token_at(dir, pos);
  scan[hidden(dir, t, d)] = scan[coef_a(dir, t, d)] * carry + scan[coef_b(dir, t, d)];
}

@compute @workgroup_size(64)
fn mix(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let t = wg.x;
  let j = lid.x;
  if (j >= MIX || t >= params.tokens) { return; }
  var acc = weights[params.off_mix_b + j];
  let row = params.off_mix_w + j * 3u * DIM;
  for (var i = 0u; i < DIM; i = i + 1u) {
    acc = acc + weights[row + i] * state[t * DIM + i]
      + weights[row + DIM + i] * scan[hidden(0u, t, i)]
      + weights[row + 2u * DIM + i] * scan[hidden(1u, t, i)];
  }
  state[m_off() + t * MIX + j] = relu(acc);
}

@compute @workgroup_size(64)
fn pool(@builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let n = params.tokens;
  let j = lid.x;
  // g = mean_t m_t
  if (j < MIX) {
    var acc = 0.0;
    for (var t = 0u; t < n; t = t + 1u) { acc = acc + state[m_off() + t * MIX + j]; }
    if (n > 0u) { acc = acc / f32(n); }
    wg_s[j] = acc;
    state[g_off() + j] = acc;
  }
  workgroupBarrier();
  // kind hidden = relu(W_k1 g + b_k1)
  var kh = 0.0;
  if (j < KH) {
    kh = weights[params.off_kind1_b + j];
    for (var i = 0u; i < MIX; i = i + 1u) { kh = kh + weights[params.off_kind1_w + j * MIX + i] * wg_s[i]; }
    kh = relu(kh);
  }
  workgroupBarrier();
  if (j < KH) { wg_s[j] = kh; }
  workgroupBarrier();
  if (j < params.kinds) {
    var acc = weights[params.off_kind2_b + j];
    for (var i = 0u; i < KH; i = i + 1u) { acc = acc + weights[params.off_kind2_w + j * KH + i] * wg_s[i]; }
    logits[n * params.labels + j] = acc;
  }
}

@compute @workgroup_size(64)
fn head(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let t = wg.x;
  let j = lid.x;
  var s = 0.0;
  if (j < MIX && t < params.tokens) {
    s = weights[params.off_head_b + j];
    let row = params.off_head_w + j * 2u * MIX;
    for (var i = 0u; i < MIX; i = i + 1u) {
      s = s + weights[row + i] * state[m_off() + t * MIX + i] + weights[row + MIX + i] * state[g_off() + i];
    }
    s = relu(s);
  }
  wg_s[j] = s;
  workgroupBarrier();
  if (j < params.labels && t < params.tokens) {
    var acc = weights[params.off_out_b + j];
    for (var i = 0u; i < MIX; i = i + 1u) { acc = acc + weights[params.off_out_w + j * MIX + i] * wg_s[i]; }
    logits[t * params.labels + j] = acc;
  }
}
