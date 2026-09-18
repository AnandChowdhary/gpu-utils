import type { FeatureRows } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/** Raw model outputs for one reference: [n, K] tag logits, [n, P] name-part logits, [T] type logits. */
export interface Logits {
  n: number;
  tags: Float32Array;
  parts: Float32Array;
  type: Float32Array;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Gated affine scan h_t = a_t * h_{t-1} + (1 - a_t) * b_t with a = σ(x Wa + ba), b = x Wb + bb.
 * Runs sequentially here; the WGSL kernel computes the same recurrence as a parallel prefix
 * scan over affine maps.
 */
function scan(
  x: Float32Array,
  n: number,
  dIn: number,
  h: number,
  wa: Float32Array,
  ba: Float32Array,
  wb: Float32Array,
  bb: Float32Array,
  reverse: boolean,
  out: Float32Array,
  outStride: number,
  outOffset: number,
): void {
  const prev = new Float64Array(h);
  for (let step = 0; step < n; step++) {
    const t = reverse ? n - 1 - step : step;
    for (let c = 0; c < h; c++) {
      let a = ba[c]!;
      let b = bb[c]!;
      for (let i = 0; i < dIn; i++) {
        const xi = x[t * dIn + i]!;
        a += xi * wa[i * h + c]!;
        b += xi * wb[i * h + c]!;
      }
      a = sigmoid(a);
      const v = a * prev[c]! + (1 - a) * b;
      prev[c] = v;
      out[t * outStride + outOffset + c] = v;
    }
  }
}

function dense(
  x: Float32Array,
  xOffset: number,
  dIn: number,
  w: Float32Array,
  b: Float32Array,
  dOut: number,
  out: Float32Array,
  outOffset: number,
  relu: boolean,
): void {
  for (let j = 0; j < dOut; j++) {
    let s = b[j]!;
    for (let i = 0; i < dIn; i++) s += x[xOffset + i]! * w[i * dOut + j]!;
    out[outOffset + j] = relu && s < 0 ? 0 : s;
  }
}

/**
 * Reference forward pass in plain TypeScript, mirroring training/gpu_cite/model.py.
 * This is the source of truth the WGSL kernels are checked against and the fallback for
 * small inputs and environments without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Logits {
  const m = model.manifest;
  const t = model.t;
  const n = features.tokens.length;
  const E = m.embed;
  const H = m.hidden;
  const C = 2 * H;
  const HEAD = m.head;
  const K = m.labels.length;
  const P = m.nameparts.length;
  const T = m.types.length;
  const tags = new Float32Array(n * K);
  const parts = new Float32Array(n * P);
  const type = new Float32Array(T);
  if (n === 0) return { n, tags, parts, type };

  // 1. summed sparse embeddings (id 0 = padding row, always zero)
  const emb = t.emb!;
  const e = new Float32Array(n * E);
  for (let i = 0; i < n; i++) {
    for (const id of features.rows[i]!) {
      if (id === 0) continue;
      for (let c = 0; c < E; c++) e[i * E + c] += emb[id * E + c]!;
    }
  }

  // 2. bidirectional scan 1 → h1 [n, C]
  const h1 = new Float32Array(n * C);
  scan(e, n, E, H, t.s1f_wa!, t.s1f_ba!, t.s1f_wb!, t.s1f_bb!, false, h1, C, 0);
  scan(e, n, E, H, t.s1b_wa!, t.s1b_ba!, t.s1b_wb!, t.s1b_bb!, true, h1, C, H);

  // 3. depthwise conv (kernel 3, zero padded) with residual: y = h1 + relu(conv(h1))
  const cw = t.conv_w!;
  const cb = t.conv_b!;
  const y = new Float32Array(n * C);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < C; c++) {
      const left = i > 0 ? h1[(i - 1) * C + c]! : 0;
      const right = i + 1 < n ? h1[(i + 1) * C + c]! : 0;
      const v = left * cw[c]! + h1[i * C + c]! * cw[C + c]! + right * cw[2 * C + c]! + cb[c]!;
      y[i * C + c] = h1[i * C + c]! + (v > 0 ? v : 0);
    }
  }

  // 4. bidirectional scan 2 → h2 [n, C]
  const h2 = new Float32Array(n * C);
  scan(y, n, C, H, t.s2f_wa!, t.s2f_ba!, t.s2f_wb!, t.s2f_bb!, false, h2, C, 0);
  scan(y, n, C, H, t.s2b_wa!, t.s2b_ba!, t.s2b_wb!, t.s2b_bb!, true, h2, C, H);

  // 5. pooled context: mean and max over tokens
  const mean = new Float32Array(C);
  const max = new Float32Array(C).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < C; c++) {
      const v = h2[i * C + c]!;
      mean[c] += v;
      if (v > max[c]!) max[c] = v;
    }
  }
  for (let c = 0; c < C; c++) mean[c] /= n;

  // 6. token head: g = relu([h2 | y | mean] W1 + b1) → tags, name parts
  const x = new Float32Array(3 * C);
  const g = new Float32Array(HEAD);
  x.set(mean, 2 * C);
  for (let i = 0; i < n; i++) {
    x.set(h2.subarray(i * C, (i + 1) * C), 0);
    x.set(y.subarray(i * C, (i + 1) * C), C);
    dense(x, 0, 3 * C, t.w1!, t.b1!, HEAD, g, 0, true);
    dense(g, 0, HEAD, t.wt!, t.bt!, K, tags, i * K, false);
    dense(g, 0, HEAD, t.wp!, t.bp!, P, parts, i * P, false);
  }

  // 7. type head: relu([mean | max] Wc + bc) Wd + bd
  const pooled = new Float32Array(2 * C);
  pooled.set(mean, 0);
  pooled.set(max, C);
  const c = new Float32Array(H);
  dense(pooled, 0, 2 * C, t.wc!, t.bc!, H, c, 0, true);
  dense(c, 0, H, t.wd!, t.bd!, T, type, 0, false);
  return { n, tags, parts, type };
}
