/**
 * CPU forward passes for the two model families, reading tensors by name from the
 * manifest written by gpu_utils_training.export.export_package. Mirrors models.py.
 */
import {
  type BiScanWeights,
  biScan,
  concatRows,
  dense,
  depthwiseConv,
  dilatedResidualBlock,
  meanPool,
  relu,
  sparseEmbed,
} from "./layers.ts";
import { type ModelManifest, tensor } from "./weights.ts";

export interface TaggerManifest extends ModelManifest {
  family: "scan" | "conv";
  featureRows: number;
  paddingId: number;
  /** Max feature ids per token; shorter rows are padded with paddingId on the GPU. */
  slots: number;
  hidden: number;
  head: number;
  tags: number;
  pooled: number;
  /** scan family */
  scanLayers?: number;
  convTaps?: number;
  /** conv family */
  embed?: number;
  dilations?: number[];
}

export interface TaggerModel {
  manifest: TaggerManifest;
  weights: Float32Array;
}

export interface TaggerOutput {
  /** `[n, tags]` logits, row-major. */
  tags: Float32Array;
  /** `[pooled]` sequence logits when the model has a pooled head. */
  pooled?: Float32Array;
}

/** Tensor names in export order; the WGSL kernels index the offset table in this order. */
export function scanTensorNames(m: TaggerManifest): string[] {
  const names = ["embed"];
  for (let l = 0; l < (m.scanLayers ?? 1); l++) {
    for (const d of ["f", "b"])
      for (const p of ["wa", "ba", "wu", "bu"]) names.push(`scan${l}.${d}.${p}`);
    names.push(`conv${l}.w`, `conv${l}.b`);
  }
  names.push("head.w", "head.b", "tags.w", "tags.b");
  if (m.pooled > 0) names.push("pool.w1", "pool.b1", "pool.w2", "pool.b2");
  return names;
}

export function convTensorNames(m: TaggerManifest): string[] {
  const names = ["embed", "proj.w", "proj.b"];
  for (let i = 0; i < (m.dilations ?? []).length; i++)
    names.push(`block${i}.w1`, `block${i}.b1`, `block${i}.w2`, `block${i}.b2`);
  names.push("head.w", "head.b", "tags.w", "tags.b");
  if (m.pooled > 0) names.push("pool.w1", "pool.b1", "pool.w2", "pool.b2");
  return names;
}

function pooledHead(
  model: TaggerModel,
  ctx: Float32Array,
  width: number,
): Float32Array | undefined {
  const m = model.manifest;
  if (!(m.pooled > 0)) return undefined;
  const W = (name: string) => tensor(model.weights, m, name);
  const p1 = relu(dense(ctx, 1, width, W("pool.w1"), W("pool.b1"), m.head));
  return dense(p1, 1, m.head, W("pool.w2"), W("pool.b2"), m.pooled);
}

/** Scan family: embed → (BiScan → residual depthwise conv)+ → mean ctx → head. */
export function scanTaggerForward(
  model: TaggerModel,
  rows: ArrayLike<ArrayLike<number>>,
): TaggerOutput {
  const m = model.manifest;
  const W = (name: string) => tensor(model.weights, m, name);
  const n = rows.length;
  const H = m.hidden;
  const C = 2 * H;
  const taps = m.convTaps ?? 5;
  let x = sparseEmbed(rows, W("embed"), H, m.paddingId);
  for (let l = 0; l < (m.scanLayers ?? 1); l++) {
    const w: BiScanWeights = {
      fWa: W(`scan${l}.f.wa`),
      fBa: W(`scan${l}.f.ba`),
      fWu: W(`scan${l}.f.wu`),
      fBu: W(`scan${l}.f.bu`),
      bWa: W(`scan${l}.b.wa`),
      bBa: W(`scan${l}.b.ba`),
      bWu: W(`scan${l}.b.wu`),
      bBu: W(`scan${l}.b.bu`),
    };
    const h = biScan(x, n, l === 0 ? H : C, H, w);
    const conv = relu(depthwiseConv(h, n, C, W(`conv${l}.w`), W(`conv${l}.b`), taps));
    for (let i = 0; i < conv.length; i++) conv[i] = h[i]! + conv[i]!;
    x = conv;
  }
  const ctx = meanPool(x, n, C);
  const g = relu(dense(concatRows(x, C, ctx, C, n), n, 2 * C, W("head.w"), W("head.b"), m.head));
  const tags = dense(g, n, m.head, W("tags.w"), W("tags.b"), m.tags);
  const pooled = pooledHead(model, ctx, C);
  return pooled ? { tags, pooled } : { tags };
}

/** Conv family: embed → proj → residual dilated blocks → per-token head (+ pooled head). */
export function convTaggerForward(
  model: TaggerModel,
  rows: ArrayLike<ArrayLike<number>>,
): TaggerOutput {
  const m = model.manifest;
  const W = (name: string) => tensor(model.weights, m, name);
  const n = rows.length;
  const E = m.embed ?? m.hidden;
  const H = m.hidden;
  let x = dense(sparseEmbed(rows, W("embed"), E, m.paddingId), n, E, W("proj.w"), W("proj.b"), H);
  (m.dilations ?? []).forEach((d, i) => {
    x = dilatedResidualBlock(
      x,
      n,
      H,
      W(`block${i}.w1`),
      W(`block${i}.b1`),
      W(`block${i}.w2`),
      W(`block${i}.b2`),
      d,
    );
  });
  const g = relu(dense(x, n, H, W("head.w"), W("head.b"), m.head));
  const tags = dense(g, n, m.head, W("tags.w"), W("tags.b"), m.tags);
  const pooled = pooledHead(model, meanPool(x, n, H), H);
  return pooled ? { tags, pooled } : { tags };
}

/** Dispatches on `manifest.family`. */
export function taggerForward(
  model: TaggerModel,
  rows: ArrayLike<ArrayLike<number>>,
): TaggerOutput {
  return model.manifest.family === "conv"
    ? convTaggerForward(model, rows)
    : scanTaggerForward(model, rows);
}
