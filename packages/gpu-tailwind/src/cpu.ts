import type { FeatureRows } from "@gpu-utils/runtime";
import { type Model, weight } from "./model.ts";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * Reference forward pass in plain TypeScript (mirrors training/gpu_tailwind/model.py):
 * summed sparse embeddings -> two gated affine scans (forward, backward) -> depthwise
 * conv3 + mean-pooled context -> two-layer head. Returns [tokens, out] logits where the
 * first `labels.length` columns are role logits and the last is the segment-boundary logit.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  const n = features.tokens.length;
  const D = model.hidden;
  const H = model.head;
  const O = model.out;
  const logits = new Float32Array(n * O);
  if (n === 0) return logits;

  const emb = weight(model, "emb");
  const waF = weight(model, "wa_f");
  const baF = weight(model, "ba_f");
  const wuF = weight(model, "wu_f");
  const buF = weight(model, "bu_f");
  const waB = weight(model, "wa_b");
  const baB = weight(model, "ba_b");
  const wuB = weight(model, "wu_b");
  const buB = weight(model, "bu_b");
  const conv = weight(model, "conv");
  const bc = weight(model, "bc");
  const w1 = weight(model, "w1");
  const wg = weight(model, "wg");
  const b1 = weight(model, "b1");
  const w2 = weight(model, "w2");
  const b2 = weight(model, "b2");

  // e[t] = sum of embeddings
  const e = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    const row = features.rows[t]!;
    for (const id of row) {
      const base = id * D;
      for (let d = 0; d < D; d++) e[t * D + d] += emb[base + d]!;
    }
  }

  // gated scans; weights are [D_in, D_out] row-major as in torch (e @ W)
  const scan = (wa: Float32Array, ba: Float32Array, wu: Float32Array, bu: Float32Array, reverse: boolean) => {
    const a = new Float32Array(n * D);
    const b = new Float32Array(n * D);
    for (let t = 0; t < n; t++) {
      for (let j = 0; j < D; j++) {
        let sa = ba[j]!;
        let su = bu[j]!;
        for (let i = 0; i < D; i++) {
          const x = e[t * D + i]!;
          sa += x * wa[i * D + j]!;
          su += x * wu[i * D + j]!;
        }
        const g = sigmoid(sa);
        a[t * D + j] = g;
        b[t * D + j] = (1 - g) * Math.tanh(su);
      }
    }
    const h = new Float32Array(n * D);
    for (let j = 0; j < D; j++) {
      let prev = 0;
      for (let k = 0; k < n; k++) {
        const t = reverse ? n - 1 - k : k;
        prev = a[t * D + j]! * prev + b[t * D + j]!;
        h[t * D + j] = prev;
      }
    }
    return h;
  };
  const hf = scan(waF, baF, wuF, buF, false);
  const hb = scan(waB, baB, wuB, buB, true);

  // x = [e ; hf ; hb]
  const W = 3 * D;
  const x = new Float32Array(n * W);
  for (let t = 0; t < n; t++) {
    for (let d = 0; d < D; d++) {
      x[t * W + d] = e[t * D + d]!;
      x[t * W + D + d] = hf[t * D + d]!;
      x[t * W + 2 * D + d] = hb[t * D + d]!;
    }
  }
  // mean-pooled context projected once
  const gctx = new Float32Array(H);
  for (let d = 0; d < W; d++) {
    let s = 0;
    for (let t = 0; t < n; t++) s += x[t * W + d]!;
    const mean = s / n;
    for (let h = 0; h < H; h++) gctx[h] += mean * wg[d * H + h]!;
  }
  // depthwise conv3 + head
  const c = new Float32Array(W);
  const z = new Float32Array(H);
  for (let t = 0; t < n; t++) {
    for (let d = 0; d < W; d++) {
      const prev = t > 0 ? x[(t - 1) * W + d]! : 0;
      const next = t < n - 1 ? x[(t + 1) * W + d]! : 0;
      c[d] = prev * conv[d]! + x[t * W + d]! * conv[W + d]! + next * conv[2 * W + d]! + bc[d]!;
    }
    for (let h = 0; h < H; h++) {
      let s = b1[h]! + gctx[h]!;
      for (let d = 0; d < W; d++) s += c[d]! * w1[d * H + h]!;
      z[h] = s > 0 ? s : 0;
    }
    for (let o = 0; o < O; o++) {
      let s = b2[o]!;
      for (let h = 0; h < H; h++) s += z[h]! * w2[h * O + o]!;
      logits[t * O + o] = s;
    }
  }
  return logits;
}
