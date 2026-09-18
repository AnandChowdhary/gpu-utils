// Canonical scan-family kernel (gpu_utils_training.models.ScanTagger / models.ts
// scanTaggerForward). Widths come from the uniform block (limits: 2*hidden <= 256,
// head <= 256, at most 4 scan layers), 64 threads per workgroup striding over channels.
//
// Passes (runtime/gpu.ts scanTaggerPasses; every entry references every binding, see touch):
//   embed          one workgroup per token       x = sparse_embed(rows)            [T, H]
//   per layer l:
//     gates{l}     one workgroup per token       a = sigmoid(x Wa + ba), b = (1-a) tanh(x Wu + bu)
//     scan{l}      one workgroup per sequence    h_t = a_t h_{t-1} + b_t, forward and backward
//     conv{l}      one workgroup per token       x = h + relu(dwconv(h))            [T, 2H]
//   pool           one workgroup per sequence    ctx = mean_t x
//   head           one workgroup per token       tags = relu([x_t | ctx] W + b) W_tags + b
//   pooled         one workgroup per sequence    pooled = relu(ctx W1 + b) W2 + b
// Only the scan pass is sequential in T; everything else is embarrassingly parallel, and no
// invocation loops more than O(T + 2H) times (Mesa's lavapipe aborts an invocation after
// ~65K loop iterations, so keep it that way).
//
// Bindings (runtime/gpu.ts runScanTagger and gpu_utils_training.kernels mirror this):
//   0 params   uniform  Params
//   1 table    u32[]    tensor offsets in ScanTagger.tensor_names() order (pool slots zero-padded)
//   2 weights  f32[]    decoded int6 weights (runtime/weights.ts decodeInt6)
//   3 rows     u32[]    [batch, max_tokens, slots] feature ids, `padding` where unused
//   4 lengths  u32[]    [batch] real token counts
//   5 scratch  f32[]    4 planes of [batch, max_tokens, 2*hidden] (X, A, B, S) then ctx [batch, 2*hidden]
//   6 logits   f32[]    [batch, max_tokens, tags] then [batch, pooled]

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
var<workgroup> sh_x: array<f32, 512>;
var<workgroup> sh_g: array<f32, 256>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn touch() -> f32 {
  return f32(params.batch) + f32(table[0]) + weights[0] + f32(rows[0]) + f32(lengths[0]) + scratch[0] + logits[0];
}
fn token_of(wg: vec3<u32>) -> u32 { return wg.x + wg.y * GRID_X; }
fn plane() -> u32 { return params.batch * params.max_tokens * 2u * params.hidden; }
fn ctx_base(seq: u32) -> u32 { return 4u * plane() + seq * 2u * params.hidden; }
fn head_base() -> u32 { return 1u + params.layers * 10u; }

@compute @workgroup_size(64)
fn embed(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let H = params.hidden;
  let C = 2u * H;
  let valid = (p % params.max_tokens) < lengths[p / params.max_tokens];
  for (var c = lid.x; c < H; c = c + WG) {
    var total = 0.0;
    if (valid) {
      for (var s = 0u; s < params.slots; s = s + 1u) {
        let row = rows[p * params.slots + s];
        if (row != params.padding) { total = total + weights[table[0] + row * H + c]; }
      }
    }
    scratch[p * C + c] = total;
  }
}

// a = sigmoid(x Wa + ba), b = (1 - a) * tanh(x Wu + bu) for both directions; identity map on padding.
fn run_gates(l: u32, wg: vec3<u32>, id: u32) {
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let H = params.hidden;
  let C = 2u * H;
  let d_in = select(C, H, l == 0u);
  let tb = 1u + l * 10u;
  let valid = (p % params.max_tokens) < lengths[p / params.max_tokens];
  for (var i = id; i < d_in; i = i + WG) { sh_x[i] = scratch[p * C + i]; }
  workgroupBarrier();
  let A = plane() + p * C;
  let B = 2u * plane() + p * C;
  for (var c = id; c < H; c = c + WG) {
    var fa = weights[table[tb + 1u] + c];
    var fu = weights[table[tb + 3u] + c];
    var ba = weights[table[tb + 5u] + c];
    var bu = weights[table[tb + 7u] + c];
    for (var i = 0u; i < d_in; i = i + 1u) {
      let x = sh_x[i];
      fa = fa + x * weights[table[tb] + i * H + c];
      fu = fu + x * weights[table[tb + 2u] + i * H + c];
      ba = ba + x * weights[table[tb + 4u] + i * H + c];
      bu = bu + x * weights[table[tb + 6u] + i * H + c];
    }
    let af = select(1.0, sigmoid(fa), valid);
    let ab = select(1.0, sigmoid(ba), valid);
    scratch[A + c] = af;
    scratch[B + c] = (1.0 - af) * tanh(fu);
    scratch[A + H + c] = ab;
    scratch[B + H + c] = (1.0 - ab) * tanh(bu);
  }
}

// h_t = a_t * h_{t-1} + b_t: forward in columns [0, H), backward in [H, 2H); one thread per column.
fn run_scan(l: u32, wg: vec3<u32>, id: u32) {
  let seq = wg.x;
  if (seq >= params.batch) { return; }
  let H = params.hidden;
  let C = 2u * H;
  let count = min(lengths[seq], params.max_tokens);
  let base = seq * params.max_tokens * C;
  let A = plane() + base;
  let B = 2u * plane() + base;
  let S = 3u * plane() + base;
  for (var c = id; c < C; c = c + WG) {
    var state = 0.0;
    for (var step = 0u; step < count; step = step + 1u) {
      let t = select(count - 1u - step, step, c < H);
      state = scratch[A + t * C + c] * state + scratch[B + t * C + c];
      scratch[S + t * C + c] = state;
    }
  }
}

// x = h + relu(dwconv(h)), zero on padding.
fn run_conv(l: u32, wg: vec3<u32>, id: u32) {
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let H = params.hidden;
  let C = 2u * H;
  let seq = p / params.max_tokens;
  let t = p % params.max_tokens;
  let count = lengths[seq];
  let tb = 1u + l * 10u;
  let taps = params.taps;
  let half = taps / 2u;
  let S = 3u * plane() + seq * params.max_tokens * C;
  for (var c = id; c < C; c = c + WG) {
    var y = 0.0;
    if (t < count) {
      var acc = weights[table[tb + 9u] + c];
      for (var k = 0u; k < taps; k = k + 1u) {
        let src = i32(t) + i32(k) - i32(half);
        if (src >= 0 && src < i32(count)) {
          acc = acc + weights[table[tb + 8u] + k * C + c] * scratch[S + u32(src) * C + c];
        }
      }
      y = scratch[S + t * C + c] + max(acc, 0.0);
    }
    scratch[p * C + c] = y;
  }
}

@compute @workgroup_size(64) fn gates0(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_gates(0u, wg, lid.x); }
@compute @workgroup_size(64) fn gates1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_gates(1u, wg, lid.x); }
@compute @workgroup_size(64) fn gates2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_gates(2u, wg, lid.x); }
@compute @workgroup_size(64) fn gates3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_gates(3u, wg, lid.x); }
@compute @workgroup_size(64) fn scan0(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_scan(0u, wg, lid.x); }
@compute @workgroup_size(64) fn scan1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_scan(1u, wg, lid.x); }
@compute @workgroup_size(64) fn scan2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_scan(2u, wg, lid.x); }
@compute @workgroup_size(64) fn scan3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_scan(3u, wg, lid.x); }
@compute @workgroup_size(64) fn conv0(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_conv(0u, wg, lid.x); }
@compute @workgroup_size(64) fn conv1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_conv(1u, wg, lid.x); }
@compute @workgroup_size(64) fn conv2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_conv(2u, wg, lid.x); }
@compute @workgroup_size(64) fn conv3(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { _ = touch(); run_conv(3u, wg, lid.x); }

// ctx = mean over real tokens of x, one workgroup per sequence.
@compute @workgroup_size(64)
fn pool(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let seq = wg.x;
  if (seq >= params.batch) { return; }
  let C = 2u * params.hidden;
  let count = min(lengths[seq], params.max_tokens);
  let base = seq * params.max_tokens * C;
  for (var c = lid.x; c < C; c = c + WG) {
    var sum = 0.0;
    for (var t = 0u; t < count; t = t + 1u) { sum = sum + scratch[base + t * C + c]; }
    scratch[ctx_base(seq) + c] = sum / max(1.0, f32(count));
  }
}

// Per-token head: g = relu([x_t | ctx] W + b); tags = g W_tags + b_tags.
@compute @workgroup_size(64)
fn head(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let p = token_of(wg);
  if (p >= params.batch * params.max_tokens) { return; }
  let seq = p / params.max_tokens;
  if ((p % params.max_tokens) >= lengths[seq]) { return; }
  let C = 2u * params.hidden;
  let head = params.head;
  let hb = head_base();
  for (var i = lid.x; i < C; i = i + WG) {
    sh_x[i] = scratch[p * C + i];
    sh_x[C + i] = scratch[ctx_base(seq) + i];
  }
  workgroupBarrier();
  for (var j = lid.x; j < head; j = j + WG) {
    var acc = weights[table[hb + 1u] + j];
    for (var i = 0u; i < 2u * C; i = i + 1u) { acc = acc + sh_x[i] * weights[table[hb] + i * head + j]; }
    sh_g[j] = max(acc, 0.0);
  }
  workgroupBarrier();
  for (var o = lid.x; o < params.tags; o = o + WG) {
    var acc = weights[table[hb + 3u] + o];
    for (var j = 0u; j < head; j = j + 1u) { acc = acc + sh_g[j] * weights[table[hb + 2u] + j * params.tags + o]; }
    logits[p * params.tags + o] = acc;
  }
}

// Pooled head, one workgroup per sequence: pooled = relu(ctx W1 + b1) W2 + b2.
@compute @workgroup_size(64)
fn pooled(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = touch();
  let seq = wg.x;
  if (seq >= params.batch || params.pooled == 0u) { return; }
  let C = 2u * params.hidden;
  let head = params.head;
  let hb = head_base();
  for (var i = lid.x; i < C; i = i + WG) { sh_x[i] = scratch[ctx_base(seq) + i]; }
  workgroupBarrier();
  for (var j = lid.x; j < head; j = j + WG) {
    var acc = weights[table[hb + 5u] + j];
    for (var i = 0u; i < C; i = i + 1u) { acc = acc + sh_x[i] * weights[table[hb + 4u] + i * head + j]; }
    sh_g[j] = max(acc, 0.0);
  }
  workgroupBarrier();
  let base = params.batch * params.max_tokens * params.tags + seq * params.pooled;
  for (var o = lid.x; o < params.pooled; o = o + WG) {
    var acc = weights[table[hb + 7u] + o];
    for (var j = 0u; j < head; j = j + 1u) { acc = acc + sh_g[j] * weights[table[hb + 6u] + j * params.pooled + o]; }
    logits[base + o] = acc;
  }
}
