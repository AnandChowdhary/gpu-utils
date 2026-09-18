// gpu-log compute kernels. One workgroup per token, one thread per hidden channel (H = 64).
// Mirrors cpu.ts exactly: embed → 5 residual dilated conv blocks (ping-pong between
// state_a / state_b) → heads. Lines are packed back to back; `line_id` keeps convolutions
// from reading across line boundaries (neighbours in another line contribute zero).

const H: u32 = 64u;
const E: u32 = 32u;
const F: u32 = 9u;
const T: u32 = 25u;
const K: u32 = 3u;
const GRID_X: u32 = 32768u;

struct Params {
  tokens: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

// offs: [embed, proj_w, proj_b, (w1, b1, w2, b2) * 5, head_h_w, head_h_b, head_tag_w, head_tag_b, head_kind_w, head_kind_b]
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> offs: array<u32>;
@group(0) @binding(2) var<storage, read> features: array<u32>;
@group(0) @binding(3) var<storage, read> line_id: array<u32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> state_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> state_b: array<f32>;
@group(0) @binding(7) var<storage, read_write> logits: array<f32>;

var<workgroup> sh_x: array<f32, 192>;  // 3 taps × H (conv input) or H (head hidden)
var<workgroup> sh_h: array<f32, 64>;
var<workgroup> sh_e: array<f32, 32>;

fn token_of(wid: vec3<u32>) -> u32 {
  return wid.x + wid.y * GRID_X;
}

// Every entry point must statically reference every binding: "auto" bind group layouts only
// contain the bindings an entry point uses, and runtime/program.ts binds the same buffer list
// to every pass.
fn touch() -> f32 {
  return f32(params.tokens) + f32(offs[0]) + f32(features[0]) + f32(line_id[0]) + weights[0]
    + state_a[0] + state_b[0] + logits[0];
}

@compute @workgroup_size(64)
fn embed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  let o = lid.x;
  if (o < E) {
    var s = 0.0;
    for (var f = 0u; f < F; f = f + 1u) {
      let row = features[p * F + f];
      s = s + weights[offs[0] + row * E + o];
    }
    sh_e[o] = s;
  }
  workgroupBarrier();
  var acc = weights[offs[2] + o];
  let pw = offs[1];
  for (var e = 0u; e < E; e = e + 1u) {
    acc = acc + sh_e[e] * weights[pw + e * H + o];
  }
  state_a[p * H + o] = acc;
}

// One residual block: dst = src + W2·relu(conv3(src)) + b2, with dilation d.
fn run_block(p: u32, o: u32, d: u32, base: u32, from_a: bool) {
  let n = params.tokens;
  let lid = line_id[p];
  // Load the three tap vectors (zero when outside the line) into shared memory.
  for (var tap = 0u; tap < 3u; tap = tap + 1u) {
    let q = i32(p) + (i32(tap) - 1) * i32(d);
    var v = 0.0;
    if (q >= 0 && u32(q) < n && line_id[u32(q)] == lid) {
      if (from_a) { v = state_a[u32(q) * H + o]; } else { v = state_b[u32(q) * H + o]; }
    }
    sh_x[tap * H + o] = v;
  }
  workgroupBarrier();
  let w1 = offs[base];
  let b1 = offs[base + 1u];
  let w2 = offs[base + 2u];
  let b2 = offs[base + 3u];
  var h = weights[b1 + o];
  for (var i = 0u; i < 3u * H; i = i + 1u) {
    h = h + sh_x[i] * weights[w1 + i * H + o];
  }
  sh_h[o] = max(h, 0.0);
  workgroupBarrier();
  var y = sh_x[H + o] + weights[b2 + o];
  for (var i = 0u; i < H; i = i + 1u) {
    y = y + sh_h[i] * weights[w2 + i * H + o];
  }
  if (from_a) { state_b[p * H + o] = y; } else { state_a[p * H + o] = y; }
}

@compute @workgroup_size(64)
fn block0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  run_block(p, lid.x, 1u, 3u, true);
}

@compute @workgroup_size(64)
fn block1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  run_block(p, lid.x, 2u, 7u, false);
}

@compute @workgroup_size(64)
fn block2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  run_block(p, lid.x, 4u, 11u, true);
}

@compute @workgroup_size(64)
fn block3(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  run_block(p, lid.x, 8u, 15u, false);
}

@compute @workgroup_size(64)
fn block4(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  run_block(p, lid.x, 16u, 19u, true);
}

// After block4 the current state is in state_b.
@compute @workgroup_size(64)
fn heads(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wid);
  if (p >= params.tokens) { return; }
  let o = lid.x;
  let x = state_b[p * H + o];
  sh_x[o] = x;
  workgroupBarrier();
  let hw = offs[23];
  var h = weights[offs[24] + o];
  for (var i = 0u; i < H; i = i + 1u) {
    h = h + sh_x[i] * weights[hw + i * H + o];
  }
  sh_h[o] = max(h, 0.0);
  workgroupBarrier();
  let W = T + K;
  if (o < T) {
    let tw = offs[25];
    var acc = weights[offs[26] + o];
    for (var i = 0u; i < H; i = i + 1u) {
      acc = acc + sh_h[i] * weights[tw + i * T + o];
    }
    logits[p * W + o] = acc;
  } else if (o < W) {
    let k = o - T;
    let kw = offs[27];
    var acc = weights[offs[28] + k];
    for (var i = 0u; i < H; i = i + 1u) {
      acc = acc + sh_x[i] * weights[kw + i * K + k];
    }
    logits[p * W + o] = acc;
  }
}
