// gpu-email compute kernels. Mirrors src/cpu.ts (the reference) exactly.
//
// Layout
//   params:  n tokens, slots per token, number of conv layers
//   meta:    u32 table of tensor offsets into `weights`:
//            [0] emb  [1] head.w [2] head.b [3] kind.w [4] kind.b [5] bio.w [6] bio.b
//            then per layer l: [7+3l] conv.w  [8+3l] conv.b  [9+3l] dilation
//   state:   (L+1) slabs of n*DIM floats; layer l reads slab l and writes slab l+1
//   logits:  n * LOGITS floats (KINDS line-kind logits, then BIO logits)
//
// Thread mapping: `embed` and `conv*` use one thread per (token, channel); `head` uses
// one thread per token. Widths are compile-time constants checked against the manifest
// in gpu.ts.

const DIM: u32 = 48u;
const HEAD: u32 = 64u;
const KINDS: u32 = 8u;
const BIO: u32 = 15u;
const LOGITS: u32 = 23u;

struct Params { n: u32, slots: u32, layers: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> meta: array<u32>;
@group(0) @binding(2) var<storage, read> features: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;

fn relu(x: f32) -> f32 { return max(x, 0.0); }

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let gid = id.x;
  if (gid >= params.n * DIM) { return; }
  let t = gid / DIM;
  let d = gid % DIM;
  let emb = meta[0];
  var acc: f32 = 0.0;
  for (var s: u32 = 0u; s < params.slots; s = s + 1u) {
    let row = features[t * params.slots + s];
    acc = acc + weights[emb + row * DIM + d];
  }
  state[gid] = acc;
}

fn conv_layer(l: u32, gid: u32) {
  if (gid >= params.n * DIM) { return; }
  let t = gid / DIM;
  let co = gid % DIM;
  let w = meta[7u + 3u * l];
  let b = meta[8u + 3u * l];
  let dil = meta[9u + 3u * l];
  let src = l * params.n * DIM;
  let dst = (l + 1u) * params.n * DIM;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < 3u; k = k + 1u) {
    let j = i32(t) + (i32(k) - 1) * i32(dil);
    if (j < 0 || j >= i32(params.n)) { continue; }
    let xBase = src + u32(j) * DIM;
    let wBase = w + k * DIM * DIM + co;
    for (var ci: u32 = 0u; ci < DIM; ci = ci + 1u) {
      acc = acc + state[xBase + ci] * weights[wBase + ci * DIM];
    }
  }
  state[dst + gid] = state[src + gid] + relu(acc + weights[b + co]);
}

@compute @workgroup_size(64) fn conv0(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(0u, id.x); }
@compute @workgroup_size(64) fn conv1(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(1u, id.x); }
@compute @workgroup_size(64) fn conv2(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(2u, id.x); }
@compute @workgroup_size(64) fn conv3(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(3u, id.x); }
@compute @workgroup_size(64) fn conv4(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(4u, id.x); }
@compute @workgroup_size(64) fn conv5(@builtin(global_invocation_id) id: vec3<u32>) { conv_layer(5u, id.x); }

@compute @workgroup_size(64)
fn head(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x;
  if (t >= params.n) { return; }
  let xBase = params.layers * params.n * DIM + t * DIM;
  let headW = meta[1];
  let headB = meta[2];
  let kindW = meta[3];
  let kindB = meta[4];
  let bioW = meta[5];
  let bioB = meta[6];
  var h: array<f32, HEAD>;
  for (var j: u32 = 0u; j < HEAD; j = j + 1u) { h[j] = weights[headB + j]; }
  for (var d: u32 = 0u; d < DIM; d = d + 1u) {
    let v = state[xBase + d];
    let wBase = headW + d * HEAD;
    for (var j: u32 = 0u; j < HEAD; j = j + 1u) { h[j] = h[j] + v * weights[wBase + j]; }
  }
  for (var j: u32 = 0u; j < HEAD; j = j + 1u) { h[j] = relu(h[j]); }
  let oBase = t * LOGITS;
  for (var c: u32 = 0u; c < KINDS; c = c + 1u) {
    var s: f32 = weights[kindB + c];
    for (var j: u32 = 0u; j < HEAD; j = j + 1u) { s = s + h[j] * weights[kindW + j * KINDS + c]; }
    logits[oBase + c] = s;
  }
  for (var c: u32 = 0u; c < BIO; c = c + 1u) {
    var s: f32 = weights[bioB + c];
    for (var j: u32 = 0u; j < HEAD; j = j + 1u) { s = s + h[j] * weights[bioW + j * BIO + c]; }
    logits[oBase + KINDS + c] = s;
  }
}
