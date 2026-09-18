// gpu-view compute kernels. Must match src/cpu.ts to 1e-4.
//
// One workgroup per phrase, one thread per hidden channel (32). Every channel is
// independent through the recurrence, so each thread walks its channel's affine scan
// h[t] = a[t] * h[t-1] + b[t] while 32 channels and every phrase in the batch run in
// parallel. Per-token state lives in a storage scratch buffer so phrase length is
// unbounded; the small head buffers live in workgroup memory.
//
// Layout of `weights` follows manifest.json tensor offsets (uniform `off`).

struct Params {
  phrases: u32,
  max_tokens: u32,
  slots: u32,
  outs: u32,
  hidden: u32,
  head_gate: u32,
  padding_row: u32,
  _pad: u32,
}

struct Offsets {
  embedding: u32, encoder_bias: u32, convolution: u32, gate_w: u32,
  gate_b: u32, cand_w: u32, cand_b: u32, combine_w: u32,
  combine_b: u32, global_w: u32, global_b: u32, hg_w: u32,
  hg_b: u32, hh_w: u32, hh_b: u32, out_w: u32,
  out_b: u32, _p1: u32, _p2: u32, _p3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> off: Offsets;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> rows: array<u32>;
@group(0) @binding(4) var<storage, read> lengths: array<u32>;
// scratch: 4 planes of [phrases, max_tokens, hidden]: emb/fwd, enc, gate/combined, cand/bwd
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> logits: array<f32>;

const HIDDEN: u32 = 32u;
const HEAD_GATE: u32 = 16u;
const HEAD_HIDDEN: u32 = 64u;
const CONV: u32 = 5u;

var<workgroup> pooled: array<f32, 32>;
var<workgroup> ctx: array<f32, 32>;
var<workgroup> joined: array<f32, 80>;
var<workgroup> hidden: array<f32, 64>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(32)
fn tag(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let seq = wg.x;
  let c = lid.x;
  if (seq >= params.phrases) { return; }
  let count = min(lengths[seq], params.max_tokens);
  let plane = params.phrases * params.max_tokens * HIDDEN;
  let base = seq * params.max_tokens * HIDDEN;
  let A = 0u;          // embedded, later forward scan
  let B = plane;       // encoded
  let C = plane * 2u;  // gate, later combined
  let D = plane * 3u;  // candidate, later backward scan

  // 1. Summed sparse embeddings.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var total = 0.0;
    let rbase = (seq * params.max_tokens + t) * params.slots;
    for (var s: u32 = 0u; s < params.slots; s = s + 1u) {
      let row = rows[rbase + s];
      if (row != params.padding_row) {
        total = total + weights[off.embedding + row * HIDDEN + c];
      }
    }
    scratch[A + base + t * HIDDEN + c] = total;
  }
  // 2. Depthwise convolution + tanh (thread c only reads channel c: no barrier needed).
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var total = weights[off.encoder_bias + c];
    for (var k: u32 = 0u; k < CONV; k = k + 1u) {
      let src = i32(t) + i32(k) - 2;
      if (src >= 0 && src < i32(count)) {
        total = total + weights[off.convolution + k * HIDDEN + c] * scratch[A + base + u32(src) * HIDDEN + c];
      }
    }
    scratch[B + base + t * HIDDEN + c] = tanh(total);
  }
  workgroupBarrier();
  // 3. Gate and candidate (reads every channel of enc).
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var g = weights[off.gate_b + c];
    var k = weights[off.cand_b + c];
    for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
      let v = scratch[B + base + t * HIDDEN + i];
      g = g + v * weights[off.gate_w + c * HIDDEN + i];
      k = k + v * weights[off.cand_w + c * HIDDEN + i];
    }
    let gv = sigmoid(g);
    scratch[C + base + t * HIDDEN + c] = gv;
    scratch[D + base + t * HIDDEN + c] = (1.0 - gv) * tanh(k);
  }
  // 4. Affine scans. Forward into A; backward overwrites D in place (own channel only).
  var state = 0.0;
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    state = scratch[C + base + t * HIDDEN + c] * state + scratch[D + base + t * HIDDEN + c];
    scratch[A + base + t * HIDDEN + c] = state;
  }
  state = 0.0;
  for (var step: u32 = 0u; step < count; step = step + 1u) {
    let t = count - 1u - step;
    state = scratch[C + base + t * HIDDEN + c] * state + scratch[D + base + t * HIDDEN + c];
    scratch[D + base + t * HIDDEN + c] = state;
  }
  workgroupBarrier();
  // 5. Combine into C, then mean-pool into a gated global context.
  var sum = 0.0;
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var mixed = weights[off.combine_b + c];
    for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
      mixed = mixed + scratch[A + base + t * HIDDEN + i] * weights[off.combine_w + c * (HIDDEN * 2u) + i];
      mixed = mixed + scratch[D + base + t * HIDDEN + i] * weights[off.combine_w + c * (HIDDEN * 2u) + HIDDEN + i];
    }
    let v = tanh(scratch[B + base + t * HIDDEN + c] + mixed);
    scratch[C + base + t * HIDDEN + c] = v;
    sum = sum + v;
  }
  pooled[c] = sum / max(1.0, f32(count));
  workgroupBarrier();
  var g2 = weights[off.global_b + c];
  for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
    g2 = g2 + pooled[i] * weights[off.global_w + c * HIDDEN + i];
  }
  ctx[c] = sigmoid(g2) * pooled[c];
  workgroupBarrier();
  // 6. Head, one token at a time so 32 threads cover the 16/64/15-wide layers.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    joined[c] = scratch[C + base + t * HIDDEN + c];
    joined[HIDDEN + c] = ctx[c];
    workgroupBarrier();
    if (c < HEAD_GATE) {
      var s = weights[off.hg_b + c];
      for (var i: u32 = 0u; i < HIDDEN * 2u; i = i + 1u) {
        s = s + joined[i] * weights[off.hg_w + c * (HIDDEN * 2u) + i];
      }
      joined[HIDDEN * 2u + c] = sigmoid(s);
    }
    workgroupBarrier();
    for (var half: u32 = 0u; half < 2u; half = half + 1u) {
      let h = half * HIDDEN + c;
      var s = weights[off.hh_b + h];
      for (var i: u32 = 0u; i < HIDDEN * 2u + HEAD_GATE; i = i + 1u) {
        s = s + joined[i] * weights[off.hh_w + h * (HIDDEN * 2u + HEAD_GATE) + i];
      }
      hidden[h] = tanh(s);
    }
    workgroupBarrier();
    if (c < params.outs) {
      var s = weights[off.out_b + c];
      for (var i: u32 = 0u; i < HEAD_HIDDEN; i = i + 1u) {
        s = s + hidden[i] * weights[off.out_w + c * HEAD_HIDDEN + i];
      }
      logits[(seq * params.max_tokens + t) * params.outs + c] = s;
    }
    workgroupBarrier();
  }
}
