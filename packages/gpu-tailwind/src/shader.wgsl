// gpu-tailwind compute kernels. Mirrors src/cpu.ts / training/gpu_tailwind/model.py exactly.
//
// Passes (dispatched in order):
//   embed       e_t = sum_f E[id_f]                          one thread per (token, channel)
//   gates       a, (1-a)*tanh(u) for both scan directions    one thread per (dir, token, channel)
//   scan_local  Hillis-Steele affine prefix scan inside      one workgroup per (chunk, dir*D + channel)
//               256-token chunks (workgroup memory)
//   scan_fixup  compose chunk carries, write h, assemble x   one thread per (dir, token, channel)
//   pool        g = mean_t x_t projected through Wg          one workgroup
//   head        conv3 + relu head + output logits            one thread per token
//
// Every entry point statically references every binding (touch) because the runtime
// builds one bind group per pipeline from its auto layout.

const D: u32 = 24u;        // embedding / scan channels
const W: u32 = 72u;        // 3 * D
const H: u32 = 32u;        // head hidden
const CHUNK: u32 = 256u;   // scan chunk length (= workgroup size)

struct Params {
  tokens: u32,
  chunks: u32,
  out: u32,
  width: u32,
  off_emb: u32,
  off_wa_f: u32,
  off_ba_f: u32,
  off_wu_f: u32,
  off_bu_f: u32,
  off_wa_b: u32,
  off_ba_b: u32,
  off_wu_b: u32,
  off_bu_b: u32,
  off_conv: u32,
  off_bc: u32,
  off_w1: u32,
  off_wg: u32,
  off_b1: u32,
  off_w2: u32,
  off_b2: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
// state: e [tokens*D] | x [tokens*W] | gctx [H]
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
// scan: a [2*tokens*D] | b [2*tokens*D] | h [2*tokens*D]; index dir*tokens*D + t*D + d
@group(0) @binding(4) var<storage, read_write> scan: array<f32>;
// logits: [tokens*out]
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;

var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;
var<workgroup> wg_t: array<f32, 256>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

fn touch() -> f32 {
  return f32(params.width) * 0.0 + f32(features[0]) * 0.0 + weights[0] * 0.0 + state[0] * 0.0 + scan[0] * 0.0 + logits[0] * 0.0;
}

fn x_off() -> u32 { return params.tokens * D; }
fn g_off() -> u32 { return params.tokens * D + params.tokens * W; }

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  if (id.x >= n * D) { return; }
  let t = id.x / D;
  let d = id.x % D;
  var s = z;
  for (var f = 0u; f < params.width; f = f + 1u) {
    let row = features[t * params.width + f];
    s = s + weights[params.off_emb + row * D + d];
  }
  state[t * D + d] = s;
}

@compute @workgroup_size(64)
fn gates(@builtin(global_invocation_id) id: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  if (id.x >= 2u * n * D) { return; }
  let dir = id.x / (n * D);
  let rem = id.x % (n * D);
  let t = rem / D;
  let j = rem % D;
  var wa = params.off_wa_f;
  var ba = params.off_ba_f;
  var wu = params.off_wu_f;
  var bu = params.off_bu_f;
  if (dir == 1u) { wa = params.off_wa_b; ba = params.off_ba_b; wu = params.off_wu_b; bu = params.off_bu_b; }
  var sa = weights[ba + j] + z;
  var su = weights[bu + j];
  for (var i = 0u; i < D; i = i + 1u) {
    let x = state[t * D + i];
    sa = sa + x * weights[wa + i * D + j];
    su = su + x * weights[wu + i * D + j];
  }
  let a = sigmoid(sa);
  let base = dir * n * D + t * D + j;
  scan[base] = a;
  scan[2u * n * D + base] = (1.0 - a) * tanh(su);
}

// token index for position p (0..n-1 in scan order) of direction dir
fn tok(dir: u32, p: u32) -> u32 {
  if (dir == 0u) { return p; }
  return params.tokens - 1u - p;
}

@compute @workgroup_size(256)
fn scan_local(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  let chunk = wid.x;
  let dir = wid.y / D;
  let d = wid.y % D;
  let p = chunk * CHUNK + lid.x;
  let nd = n * D;
  let valid = p < n;
  var a = 1.0;
  var b = z;
  if (valid) {
    let base = dir * nd + tok(dir, p) * D + d;
    a = scan[base];
    b = scan[2u * nd + base];
  }
  wg_a[lid.x] = a;
  wg_b[lid.x] = b;
  workgroupBarrier();
  for (var off = 1u; off < CHUNK; off = off * 2u) {
    var pa = 1.0;
    var pb = 0.0;
    if (lid.x >= off) {
      pa = wg_a[lid.x - off];
      pb = wg_b[lid.x - off];
    }
    workgroupBarrier();
    // apply prefix (pa,pb) first, then local (a,b): h -> a*(pa*h + pb) + b
    let na = pa * a;
    let nb = a * pb + b;
    a = na;
    b = nb;
    wg_a[lid.x] = a;
    wg_b[lid.x] = b;
    workgroupBarrier();
  }
  if (valid) {
    let base = dir * nd + tok(dir, p) * D + d;
    scan[base] = a;
    scan[2u * nd + base] = b;
  }
}

@compute @workgroup_size(64)
fn scan_fixup(@builtin(global_invocation_id) id: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  if (id.x >= 2u * n * D) { return; }
  let dir = id.x / (n * D);
  let rem = id.x % (n * D);
  let p = rem / D;
  let d = rem % D;
  let nd = n * D;
  let chunk = p / CHUNK;
  // carry = h at the end of the previous chunks (sequential over chunks)
  var carry = z;
  for (var c = 0u; c < chunk; c = c + 1u) {
    let last = min(c * CHUNK + CHUNK - 1u, n - 1u);
    let base = dir * nd + tok(dir, last) * D + d;
    carry = scan[base] * carry + scan[2u * nd + base];
  }
  let t = tok(dir, p);
  let base = dir * nd + t * D + d;
  let h = scan[base] * carry + scan[2u * nd + base];
  scan[4u * nd + base] = h;
  // assemble x = [e ; hf ; hb]
  state[x_off() + t * W + (dir + 1u) * D + d] = h;
  if (dir == 0u) {
    state[x_off() + t * W + d] = state[t * D + d];
  }
}

@compute @workgroup_size(64)
fn pool(@builtin(local_invocation_id) lid: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  // mean over tokens for channels lid.x and lid.x + 64 (W = 72 > 64)
  for (var d = lid.x; d < W; d = d + 64u) {
    var s = z;
    for (var t = 0u; t < n; t = t + 1u) {
      s = s + state[x_off() + t * W + d];
    }
    wg_t[d] = s / f32(n);
  }
  workgroupBarrier();
  if (lid.x < H) {
    var s = 0.0;
    for (var d = 0u; d < W; d = d + 1u) {
      s = s + wg_t[d] * weights[params.off_wg + d * H + lid.x];
    }
    state[g_off() + lid.x] = s;
  }
}

@compute @workgroup_size(64)
fn head(@builtin(global_invocation_id) id: vec3<u32>) {
  let z = touch();
  let n = params.tokens;
  let t = id.x;
  if (t >= n) { return; }
  var c: array<f32, 72>;
  for (var d = 0u; d < W; d = d + 1u) {
    var prev = 0.0;
    var next = 0.0;
    if (t > 0u) { prev = state[x_off() + (t - 1u) * W + d]; }
    if (t + 1u < n) { next = state[x_off() + (t + 1u) * W + d]; }
    c[d] = prev * weights[params.off_conv + d] + state[x_off() + t * W + d] * weights[params.off_conv + W + d] + next * weights[params.off_conv + 2u * W + d] + weights[params.off_bc + d];
  }
  var zh: array<f32, 32>;
  for (var h = 0u; h < H; h = h + 1u) {
    var s = weights[params.off_b1 + h] + state[g_off() + h] + z;
    for (var d = 0u; d < W; d = d + 1u) {
      s = s + c[d] * weights[params.off_w1 + d * H + h];
    }
    zh[h] = max(s, 0.0);
  }
  for (var o = 0u; o < params.out; o = o + 1u) {
    var s = weights[params.off_b2 + o];
    for (var h = 0u; h < H; h = h + 1u) {
      s = s + zh[h] * weights[params.off_w2 + h * params.out + o];
    }
    logits[t * params.out + o] = s;
  }
}
