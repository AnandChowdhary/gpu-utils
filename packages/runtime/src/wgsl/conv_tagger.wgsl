// Canonical conv-family kernel (gpu_utils_training.models.ConvTagger / models.ts
// convTaggerForward). Tiled per token: one workgroup per token, 64 threads striding over
// channels, the three tap vectors staged in workgroup memory. Widths come from the uniform
// block (limits: embed <= 256, hidden <= 256, head <= 256, at most 8 blocks).
//
// Bindings (runtime/gpu.ts runConvTagger and gpu_utils_training.kernels mirror this):
//   0 params   uniform  Params
//   1 table    u32[]    tensor offsets in ConvTagger.tensor_names() order (pool slots zero-padded
//                       to 8 head entries when pooled == 0), then the block dilations
//   2 weights  f32[]    decoded int6 weights
//   3 rows     u32[]    [batch, max_tokens, slots] feature ids, `padding` where unused
//   4 lengths  u32[]    [batch] real token counts
//   5 scratch  f32[]    2 planes of [batch, max_tokens, hidden] (ping-pong between blocks)
//   6 logits   f32[]    [batch, max_tokens, tags] then [batch, pooled]
//
// Passes: embed, block0..block{layers-1}, head, pool (pool dispatched once per sequence).
// Every entry point references every binding (see touch) because runtime/program.ts
// builds one bind group per pipeline from the same buffer list. No invocation loops more
// than O(T + 3*hidden) times (Mesa's lavapipe aborts an invocation after ~65K iterations).

struct Params {
  batch: u32, max_tokens: u32, slots: u32, padding: u32,
  embed: u32, hidden: u32, head: u32, tags: u32,
  pooled: u32, layers: u32, taps: u32, p0: u32,
  p1: u32, p2: u32, p3: u32, p4: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
@group(0) @binding(4) var<storage, read> lengths: array<u32>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> logits: array<f32>;

const WG: u32 = 64u;
const GRID_X: u32 = 32768u;
var<workgroup> sh_x: array<f32, 768>;   // 3 taps x hidden, or the head input
var<workgroup> sh_h: array<f32, 256>;

fn touch() -> f32 {
  return f32(params.batch) + f32(table[0]) + weights[0] + f32(rows[0]) + f32(lengths[0]) + scratch[0] + logits[0];
}

fn token_of(wg: vec3<u32>) -> u32 { return wg.x + wg.y * GRID_X; }
fn head_base() -> u32 { return 3u + params.layers * 4u; }

@compute @workgroup_size(64)
fn embed(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let seq = p / params.max_tokens;
  let t = p % params.max_tokens;
  let H = params.hidden;
  let E = params.embed;
  let valid = t < lengths[seq];
  for (var e = lid.x; e < E; e = e + WG) {
    var total = 0.0;
    if (valid) {
      for (var s = 0u; s < params.slots; s = s + 1u) {
        let row = rows[p * params.slots + s];
        if (row != params.padding) { total = total + weights[table[0] + row * E + e]; }
      }
    }
    sh_x[e] = total;
  }
  workgroupBarrier();
  for (var o = lid.x; o < H; o = o + WG) {
    var acc = 0.0;
    if (valid) {
      acc = weights[table[2] + o];
      for (var e = 0u; e < E; e = e + 1u) { acc = acc + sh_x[e] * weights[table[1] + e * H + o]; }
    }
    scratch[p * H + o] = acc;
  }
}

// One residual block: dst = src + relu(conv3_d(src)) W2 + b2, zero outside the sequence.
fn run_block(l: u32, wg: vec3<u32>, id: u32) {
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let seq = p / params.max_tokens;
  let t = p % params.max_tokens;
  let H = params.hidden;
  let count = lengths[seq];
  let plane = params.batch * params.max_tokens * H;
  let src = (l % 2u) * plane;
  let dst = ((l + 1u) % 2u) * plane;
  let tb = 3u + l * 4u;
  let d = table[head_base() + 8u + l];
  for (var tap = 0u; tap < 3u; tap = tap + 1u) {
    let q = i32(t) + (i32(tap) - 1) * i32(d);
    let inside = q >= 0 && u32(q) < count;
    for (var c = id; c < H; c = c + WG) {
      var v = 0.0;
      if (inside) { v = scratch[src + (seq * params.max_tokens + u32(q)) * H + c]; }
      sh_x[tap * H + c] = v;
    }
  }
  workgroupBarrier();
  for (var o = id; o < H; o = o + WG) {
    var h = weights[table[tb + 1u] + o];
    for (var i = 0u; i < 3u * H; i = i + 1u) { h = h + sh_x[i] * weights[table[tb] + i * H + o]; }
    sh_h[o] = max(h, 0.0);
  }
  workgroupBarrier();
  for (var o = id; o < H; o = o + WG) {
    var y = 0.0;
    if (t < count) {
      y = sh_x[H + o] + weights[table[tb + 3u] + o];
      for (var i = 0u; i < H; i = i + 1u) { y = y + sh_h[i] * weights[table[tb + 2u] + i * H + o]; }
    }
    scratch[dst + p * H + o] = y;
  }
}

@compute @workgroup_size(64) fn block0(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(0u, wg, lid.x); }
@compute @workgroup_size(64) fn block1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(1u, wg, lid.x); }
@compute @workgroup_size(64) fn block2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(2u, wg, lid.x); }
@compute @workgroup_size(64) fn block3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(3u, wg, lid.x); }
@compute @workgroup_size(64) fn block4(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(4u, wg, lid.x); }
@compute @workgroup_size(64) fn block5(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(5u, wg, lid.x); }
@compute @workgroup_size(64) fn block6(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(6u, wg, lid.x); }
@compute @workgroup_size(64) fn block7(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_block(7u, wg, lid.x); }

// Per-token head: g = relu(x W_head + b); tags = g W_tags + b.
@compute @workgroup_size(64)
fn head(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let seq = p / params.max_tokens;
  let t = p % params.max_tokens;
  if (t >= lengths[seq]) { return; }
  let H = params.hidden;
  let head = params.head;
  let hb = head_base();
  let src = (params.layers % 2u) * params.batch * params.max_tokens * H;
  for (var c = lid.x; c < H; c = c + WG) { sh_x[c] = scratch[src + p * H + c]; }
  workgroupBarrier();
  for (var j = lid.x; j < head; j = j + WG) {
    var acc = weights[table[hb + 1u] + j];
    for (var i = 0u; i < H; i = i + 1u) { acc = acc + sh_x[i] * weights[table[hb] + i * head + j]; }
    sh_h[j] = max(acc, 0.0);
  }
  workgroupBarrier();
  for (var o = lid.x; o < params.tags; o = o + WG) {
    var acc = weights[table[hb + 3u] + o];
    for (var j = 0u; j < head; j = j + 1u) { acc = acc + sh_h[j] * weights[table[hb + 2u] + j * params.tags + o]; }
    logits[p * params.tags + o] = acc;
  }
}

// Pooled head, one workgroup per sequence: ctx = mean_t x; pooled = relu(ctx W1 + b1) W2 + b2.
@compute @workgroup_size(64)
fn pool(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let seq = wg.x;
  if (seq >= params.batch || params.pooled == 0u) { return; }
  let H = params.hidden;
  let head = params.head;
  let hb = head_base();
  let count = min(lengths[seq], params.max_tokens);
  let src = (params.layers % 2u) * params.batch * params.max_tokens * H + seq * params.max_tokens * H;
  for (var c = lid.x; c < H; c = c + WG) {
    var sum = 0.0;
    for (var t = 0u; t < count; t = t + 1u) { sum = sum + scratch[src + t * H + c]; }
    sh_x[c] = sum / max(1.0, f32(count));
  }
  workgroupBarrier();
  for (var j = lid.x; j < head; j = j + WG) {
    var acc = weights[table[hb + 5u] + j];
    for (var i = 0u; i < H; i = i + 1u) { acc = acc + sh_x[i] * weights[table[hb + 4u] + i * head + j]; }
    sh_h[j] = max(acc, 0.0);
  }
  workgroupBarrier();
  let base = params.batch * params.max_tokens * params.tags + seq * params.pooled;
  for (var o = lid.x; o < params.pooled; o = o + WG) {
    var acc = weights[table[hb + 7u] + o];
    for (var j = 0u; j < head; j = j + 1u) { acc = acc + sh_h[j] * weights[table[hb + 6u] + j * params.pooled + o]; }
    logits[base + o] = acc;
  }
}
