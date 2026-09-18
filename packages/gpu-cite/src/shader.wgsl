// gpu-cite compute kernels. Mirrors src/cpu.ts / training/gpu_cite/model.py op for op.
//
// A batch of S references is laid out as one flat token stream of N tokens; info holds the
// tensor offsets into `weights`, the state offsets into `state`, a per-sequence (start, len)
// table and a token→sequence map. Kernels run in order:
//   embed → gates1 → scan1 → conv → gates2 → scan2 → pool → head_hidden → head_out
//         → type_hidden → type_out
// The gated affine scans use a workgroup-level Hillis–Steele parallel prefix scan over
// affine maps (a, b) with a carry between 256-token chunks.

struct Params {
  n: u32,     // total tokens
  s: u32,     // sequences
  e: u32,     // embedding dim
  h: u32,     // scan hidden per direction (C = 2h)
  head: u32,  // token head width
  k: u32,     // tag labels
  p: u32,     // name-part labels
  t: u32,     // document types
  width: u32, // feature ids per token
  _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> info: array<u32>;
@group(0) @binding(2) var<storage, read> features: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;

// info layout (see gpu.ts buildMeta)
const M_EMB: u32 = 0u;
const M_S1F_WA: u32 = 1u; const M_S1F_BA: u32 = 2u; const M_S1F_WB: u32 = 3u; const M_S1F_BB: u32 = 4u;
const M_S1B_WA: u32 = 5u; const M_S1B_BA: u32 = 6u; const M_S1B_WB: u32 = 7u; const M_S1B_BB: u32 = 8u;
const M_CONV_W: u32 = 9u; const M_CONV_B: u32 = 10u;
const M_S2F_WA: u32 = 11u; const M_S2F_BA: u32 = 12u; const M_S2F_WB: u32 = 13u; const M_S2F_BB: u32 = 14u;
const M_S2B_WA: u32 = 15u; const M_S2B_BA: u32 = 16u; const M_S2B_WB: u32 = 17u; const M_S2B_BB: u32 = 18u;
const M_W1: u32 = 19u; const M_B1: u32 = 20u; const M_WT: u32 = 21u; const M_BT: u32 = 22u;
const M_WP: u32 = 23u; const M_BP: u32 = 24u; const M_WC: u32 = 25u; const M_BC: u32 = 26u;
const M_WD: u32 = 27u; const M_BD: u32 = 28u;
const ST_E: u32 = 29u; const ST_G1: u32 = 30u; const ST_H1: u32 = 31u; const ST_Y: u32 = 32u;
const ST_G2: u32 = 33u; const ST_H2: u32 = 34u; const ST_POOL: u32 = 35u; const ST_G: u32 = 36u;
const ST_C: u32 = 37u;
const M_SEQ: u32 = 38u; // then 2*s entries (start, len), then n token→sequence entries

const CHUNK: u32 = 256u;

// Every entry point must statically reference every binding: WebGPU "auto" bind group
// layouts are exclusive to their pipeline and runtime/program.ts builds one bind group per
// pass from the full binding list. The compiler folds this away.
fn touch() -> f32 {
  return f32(params.n) + f32(info[0]) + f32(features[0]) + weights[0] + state[0] + logits[0];
}

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn relu(x: f32) -> f32 { return max(x, 0.0); }
fn tok_seq(t: u32) -> u32 { return info[M_SEQ + 2u * params.s + t]; }
fn seq_start(s: u32) -> u32 { return info[M_SEQ + 2u * s]; }
fn seq_len(s: u32) -> u32 { return info[M_SEQ + 2u * s + 1u]; }

// 1. summed sparse embeddings: state.e[t, c] = Σ emb[row, c]  (id 0 = padding row)
@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let E = params.e;
  if (id.x >= params.n * E) { return; }
  let t = id.x / E;
  let c = id.x % E;
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < params.width; i++) {
    let row = features[t * params.width + i];
    if (row != 0u) { acc += weights[info[M_EMB] + row * E + c]; }
  }
  state[info[ST_E] + t * E + c] = acc;
}

// Gate pre-pass: a = σ(x·Wa + ba), b = x·Wb + bb  → (A, B) = (a, (1-a)·b) per token/dir/channel.
fn gates(gid: u32, din: u32, xoff: u32, wa_f: u32, ba_f: u32, wb_f: u32, bb_f: u32,
         wa_b: u32, ba_b: u32, wb_b: u32, bb_b: u32, out: u32) {
  let H = params.h;
  let C = 2u * H;
  if (gid >= params.n * C) { return; }
  let t = gid / C;
  let rem = gid % C;
  let dir = rem / H;
  let c = rem % H;
  var wa = info[wa_f]; var ba = info[ba_f]; var wb = info[wb_f]; var bb = info[bb_f];
  if (dir == 1u) { wa = info[wa_b]; ba = info[ba_b]; wb = info[wb_b]; bb = info[bb_b]; }
  var a: f32 = weights[ba + c];
  var b: f32 = weights[bb + c];
  for (var i: u32 = 0u; i < din; i++) {
    let x = state[xoff + t * din + i];
    a += x * weights[wa + i * H + c];
    b += x * weights[wb + i * H + c];
  }
  a = sigmoid(a);
  let o = info[out] + (t * C + rem) * 2u;
  state[o] = a;
  state[o + 1u] = (1.0 - a) * b;
}

@compute @workgroup_size(64)
fn gates1(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  gates(id.x, params.e, info[ST_E], M_S1F_WA, M_S1F_BA, M_S1F_WB, M_S1F_BB,
        M_S1B_WA, M_S1B_BA, M_S1B_WB, M_S1B_BB, ST_G1);
}

@compute @workgroup_size(64)
fn gates2(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  gates(id.x, 2u * params.h, info[ST_Y], M_S2F_WA, M_S2F_BA, M_S2F_WB, M_S2F_BB,
        M_S2B_WA, M_S2B_BA, M_S2B_WB, M_S2B_BB, ST_G2);
}

// Parallel prefix scan of affine maps: one workgroup per (sequence, direction, channel),
// 256 tokens per chunk, carry between chunks. h_t = A_t·h_{t-1} + B_t with h_0 = 0.
var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;

fn scan_layer(wg: u32, lid: u32, gates_off: u32, out_off: u32) {
  let H = params.h;
  let C = 2u * H;
  let s = wg / C;
  let rem = wg % C;
  let dir = rem / H;
  let start = seq_start(s);
  let len = seq_len(s);
  var carry: f32 = 0.0;
  var chunk: u32 = 0u;
  loop {
    if (chunk >= len) { break; }
    let j = chunk + lid;
    var a: f32 = 1.0;
    var b: f32 = 0.0;
    var t: u32 = 0u;
    if (j < len) {
      t = select(start + len - 1u - j, start + j, dir == 0u);
      let gi = gates_off + (t * C + rem) * 2u;
      a = state[gi];
      b = state[gi + 1u];
    }
    sa[lid] = a;
    sb[lid] = b;
    workgroupBarrier();
    var stride: u32 = 1u;
    loop {
      if (stride >= CHUNK) { break; }
      var na = sa[lid];
      var nb = sb[lid];
      if (lid >= stride) {
        let pa = sa[lid - stride];
        let pb = sb[lid - stride];
        nb = na * pb + nb;
        na = na * pa;
      }
      workgroupBarrier();
      sa[lid] = na;
      sb[lid] = nb;
      workgroupBarrier();
      stride = stride * 2u;
    }
    if (j < len) {
      state[out_off + t * C + rem] = sa[lid] * carry + sb[lid];
    }
    let last = min(CHUNK - 1u, len - chunk - 1u);
    carry = sa[last] * carry + sb[last];
    workgroupBarrier();
    chunk = chunk + CHUNK;
  }
}

@compute @workgroup_size(256)
fn scan1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  scan_layer(wg.x, lid.x, info[ST_G1], info[ST_H1]);
}

@compute @workgroup_size(256)
fn scan2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  scan_layer(wg.x, lid.x, info[ST_G2], info[ST_H2]);
}

// Depthwise conv (kernel 3, zero padded at sequence edges) with residual: y = h1 + relu(conv).
@compute @workgroup_size(64)
fn conv(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let C = 2u * params.h;
  if (id.x >= params.n * C) { return; }
  let t = id.x / C;
  let c = id.x % C;
  let s = tok_seq(t);
  let start = seq_start(s);
  let end = start + seq_len(s);
  let h1 = info[ST_H1];
  let cw = info[M_CONV_W];
  var left: f32 = 0.0;
  var right: f32 = 0.0;
  if (t > start) { left = state[h1 + (t - 1u) * C + c]; }
  if (t + 1u < end) { right = state[h1 + (t + 1u) * C + c]; }
  let mid = state[h1 + t * C + c];
  let v = left * weights[cw + c] + mid * weights[cw + C + c] + right * weights[cw + 2u * C + c]
        + weights[info[M_CONV_B] + c];
  state[info[ST_Y] + t * C + c] = mid + relu(v);
}

// Mean and max of h2 over each sequence: pool[s, c] and pool[s, C + c].
@compute @workgroup_size(64)
fn pool(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let C = 2u * params.h;
  if (id.x >= params.s * C) { return; }
  let s = id.x / C;
  let c = id.x % C;
  let start = seq_start(s);
  let len = seq_len(s);
  let h2 = info[ST_H2];
  var sum: f32 = 0.0;
  var mx: f32 = -3.4e38;
  for (var t: u32 = start; t < start + len; t++) {
    let v = state[h2 + t * C + c];
    sum += v;
    mx = max(mx, v);
  }
  let o = info[ST_POOL] + s * 2u * C;
  state[o + c] = sum / f32(max(len, 1u));
  state[o + C + c] = mx;
}

// g[t, j] = relu([h2_t | y_t | mean_s] · W1 + b1)
@compute @workgroup_size(64)
fn head_hidden(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let HEAD = params.head;
  if (id.x >= params.n * HEAD) { return; }
  let t = id.x / HEAD;
  let j = id.x % HEAD;
  let C = 2u * params.h;
  let w1 = info[M_W1];
  let h2 = info[ST_H2] + t * C;
  let y = info[ST_Y] + t * C;
  let mean = info[ST_POOL] + tok_seq(t) * 2u * C;
  var acc: f32 = weights[info[M_B1] + j];
  for (var i: u32 = 0u; i < C; i++) { acc += state[h2 + i] * weights[w1 + i * HEAD + j]; }
  for (var i: u32 = 0u; i < C; i++) { acc += state[y + i] * weights[w1 + (C + i) * HEAD + j]; }
  for (var i: u32 = 0u; i < C; i++) { acc += state[mean + i] * weights[w1 + (2u * C + i) * HEAD + j]; }
  state[info[ST_G] + t * HEAD + j] = relu(acc);
}

// tags[t, k] = g_t · Wt + bt ; parts[t, p] = g_t · Wp + bp  (parts stored after all tags)
@compute @workgroup_size(64)
fn head_out(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let K = params.k;
  let P = params.p;
  let KP = K + P;
  if (id.x >= params.n * KP) { return; }
  let t = id.x / KP;
  let o = id.x % KP;
  let HEAD = params.head;
  let g = info[ST_G] + t * HEAD;
  if (o < K) {
    var acc: f32 = weights[info[M_BT] + o];
    let wt = info[M_WT];
    for (var j: u32 = 0u; j < HEAD; j++) { acc += state[g + j] * weights[wt + j * K + o]; }
    logits[t * K + o] = acc;
  } else {
    let p = o - K;
    var acc: f32 = weights[info[M_BP] + p];
    let wp = info[M_WP];
    for (var j: u32 = 0u; j < HEAD; j++) { acc += state[g + j] * weights[wp + j * P + p]; }
    logits[params.n * K + t * P + p] = acc;
  }
}

// c[s, j] = relu([mean_s | max_s] · Wc + bc)
@compute @workgroup_size(64)
fn type_hidden(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let H = params.h;
  if (id.x >= params.s * H) { return; }
  let s = id.x / H;
  let j = id.x % H;
  let C = 2u * H;
  let pooled = info[ST_POOL] + s * 2u * C;
  let wc = info[M_WC];
  var acc: f32 = weights[info[M_BC] + j];
  for (var i: u32 = 0u; i < 2u * C; i++) { acc += state[pooled + i] * weights[wc + i * H + j]; }
  state[info[ST_C] + s * H + j] = relu(acc);
}

// type[s, ty] = c_s · Wd + bd  (stored after tags and parts)
@compute @workgroup_size(64)
fn type_out(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = touch();
  let T = params.t;
  if (id.x >= params.s * T) { return; }
  let s = id.x / T;
  let ty = id.x % T;
  let H = params.h;
  let c = info[ST_C] + s * H;
  let wd = info[M_WD];
  var acc: f32 = weights[info[M_BD] + ty];
  for (var j: u32 = 0u; j < H; j++) { acc += state[c + j] * weights[wd + j * T + ty]; }
  logits[params.n * (params.k + params.p) + s * T + ty] = acc;
}
