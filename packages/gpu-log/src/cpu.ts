import type { Model } from "./model.ts";

/**
 * A batch of lines packed into one flat token sequence. `lineId[t]` tells the convolutions
 * where line boundaries are: neighbours in another line read as zero vectors, exactly like
 * the zero-padding used in training. Both forward passes consume this layout.
 */
export interface Batch {
  /** Number of tokens in the batch. */
  n: number;
  /** [n * featureCount] sparse embedding rows. */
  features: Uint32Array;
  /** [n] index of the line each token belongs to. */
  lineId: Uint32Array;
  /** [lines + 1] token offsets of each line. */
  offsets: number[];
}

/**
 * Reference forward pass in plain TypeScript, mirroring training/gpu_log/model.py
 * (`forward_numpy`). Returns [n, tags + kinds] logits: the tag logits of each token followed
 * by its per-token line-kind logits (averaged per line by the decoder).
 */
export function forwardCpu(model: Model, batch: Batch): Float32Array {
  const { n, features, lineId } = batch;
  const H = model.hidden;
  const E = model.embedDim;
  const F = model.featureCount;
  const T = model.tags;
  const K = model.kinds;
  const out = new Float32Array(n * (T + K));
  if (n === 0) return out;

  // Embedding sum + projection.
  let x = new Float32Array(n * H);
  const s = new Float32Array(E);
  const { embed, projW, projB } = model;
  for (let p = 0; p < n; p++) {
    s.fill(0);
    for (let f = 0; f < F; f++) {
      const row = features[p * F + f]! * E;
      for (let e = 0; e < E; e++) s[e]! += embed[row + e]!;
    }
    const xo = p * H;
    for (let o = 0; o < H; o++) x[xo + o] = projB[o]!;
    for (let e = 0; e < E; e++) {
      const v = s[e]!;
      const wo = e * H;
      for (let o = 0; o < H; o++) x[xo + o]! += v * projW[wo + o]!;
    }
  }

  // Residual dilated blocks.
  let y = new Float32Array(n * H);
  const h = new Float32Array(H);
  for (const blk of model.blocks) {
    const d = blk.dilation;
    const { w1, b1, w2, b2 } = blk;
    for (let p = 0; p < n; p++) {
      for (let o = 0; o < H; o++) h[o] = b1[o]!;
      const lid = lineId[p]!;
      for (let tap = 0; tap < 3; tap++) {
        const q = p + (tap - 1) * d;
        if (q < 0 || q >= n || lineId[q] !== lid) continue;
        const xq = q * H;
        const wt = tap * H * H;
        for (let i = 0; i < H; i++) {
          const v = x[xq + i]!;
          if (v === 0) continue;
          const wo = wt + i * H;
          for (let o = 0; o < H; o++) h[o]! += v * w1[wo + o]!;
        }
      }
      const yo = p * H;
      for (let o = 0; o < H; o++) y[yo + o] = x[yo + o]! + b2[o]!;
      for (let i = 0; i < H; i++) {
        const v = h[i]!;
        if (v <= 0) continue;
        const wo = i * H;
        for (let o = 0; o < H; o++) y[yo + o]! += v * w2[wo + o]!;
      }
    }
    const tmp = x;
    x = y;
    y = tmp;
  }

  // Heads.
  const { headHW, headHB, headTagW, headTagB, headKindW, headKindB } = model;
  const hh = new Float32Array(H);
  const W = T + K;
  for (let p = 0; p < n; p++) {
    const xo = p * H;
    for (let o = 0; o < H; o++) hh[o] = headHB[o]!;
    for (let i = 0; i < H; i++) {
      const v = x[xo + i]!;
      const wo = i * H;
      for (let o = 0; o < H; o++) hh[o]! += v * headHW[wo + o]!;
    }
    const lo = p * W;
    for (let t = 0; t < T; t++) out[lo + t] = headTagB[t]!;
    for (let i = 0; i < H; i++) {
      const v = hh[i]!;
      if (v <= 0) continue;
      const wo = i * T;
      for (let t = 0; t < T; t++) out[lo + t]! += v * headTagW[wo + t]!;
    }
    for (let k = 0; k < K; k++) out[lo + T + k] = headKindB[k]!;
    for (let i = 0; i < H; i++) {
      const v = x[xo + i]!;
      const wo = i * K;
      for (let k = 0; k < K; k++) out[lo + T + k]! += v * headKindW[wo + k]!;
    }
  }
  return out;
}
