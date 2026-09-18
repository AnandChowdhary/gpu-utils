import type { FeatureRows } from "@gpu-utils/runtime";
import { logitWidth, type Model, weight } from "./model.ts";

/**
 * Reference forward pass in plain TypeScript, mirroring training/gpu_email/model.py:
 *
 *   x0      = sum over slots of emb[id]                      [n, D]
 *   x(l+1)  = x(l) + relu(conv1d_k3_dil(x(l)))               residual, zero padded
 *   h       = relu(x(L) · head.w + head.b)                   [n, H]
 *   logits  = [h · kind.w + kind.b | h · bio.w + bio.b]      [n, K + B]
 *
 * Returns one row of K + B logits per token (line kinds, then BIO labels). This is the
 * source of truth the WGSL kernels are checked against and the fallback without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  const n = features.tokens.length;
  const D = model.manifest.hidden;
  const H = model.manifest.head;
  const K = model.manifest.labels.length;
  const B = model.manifest.fields.length;
  const W = logitWidth(model);
  const out = new Float32Array(n * W);
  if (n === 0) return out;

  const emb = weight(model, "emb");
  let x = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    const row = features.rows[i]!;
    const base = i * D;
    for (const id of row) {
      const off = id * D;
      for (let d = 0; d < D; d++) x[base + d] = x[base + d]! + emb[off + d]!;
    }
  }

  const dilations = model.manifest.dilations;
  for (let l = 0; l < dilations.length; l++) {
    const dil = dilations[l]!;
    const w = weight(model, `conv${l}.w`); // [3, D_in, D_out]
    const b = weight(model, `conv${l}.b`);
    const y = new Float32Array(n * D);
    for (let i = 0; i < n; i++) {
      const yBase = i * D;
      for (let k = 0; k < 3; k++) {
        const j = i + (k - 1) * dil;
        if (j < 0 || j >= n) continue;
        const xBase = j * D;
        for (let ci = 0; ci < D; ci++) {
          const v = x[xBase + ci]!;
          if (v === 0) continue;
          const wBase = (k * D + ci) * D;
          for (let co = 0; co < D; co++) y[yBase + co] = y[yBase + co]! + v * w[wBase + co]!;
        }
      }
    }
    const next = new Float32Array(n * D);
    for (let i = 0; i < n; i++) {
      const base = i * D;
      for (let d = 0; d < D; d++) {
        const a = y[base + d]! + b[d]!;
        next[base + d] = x[base + d]! + (a > 0 ? a : 0);
      }
    }
    x = next;
  }

  const headW = weight(model, "head.w"); // [D, H]
  const headB = weight(model, "head.b");
  const kindW = weight(model, "kind.w"); // [H, K]
  const kindB = weight(model, "kind.b");
  const bioW = weight(model, "bio.w"); // [H, B]
  const bioB = weight(model, "bio.b");
  const h = new Float32Array(H);
  for (let i = 0; i < n; i++) {
    const xBase = i * D;
    for (let j = 0; j < H; j++) h[j] = headB[j]!;
    for (let d = 0; d < D; d++) {
      const v = x[xBase + d]!;
      const wBase = d * H;
      for (let j = 0; j < H; j++) h[j] = h[j]! + v * headW[wBase + j]!;
    }
    for (let j = 0; j < H; j++) if (h[j]! < 0) h[j] = 0;
    const oBase = i * W;
    for (let c = 0; c < K; c++) {
      let s = kindB[c]!;
      for (let j = 0; j < H; j++) s += h[j]! * kindW[j * K + c]!;
      out[oBase + c] = s;
    }
    for (let c = 0; c < B; c++) {
      let s = bioB[c]!;
      for (let j = 0; j < H; j++) s += h[j]! * bioW[j * B + c]!;
      out[oBase + K + c] = s;
    }
  }
  return out;
}
