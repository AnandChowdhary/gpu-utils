/**
 * Batch packing for the canonical kernels. Mirrors gpu_utils_training/batch.py and
 * kernels.py: `rows [batch, maxTokens, slots]` (u32, `paddingId` where unused),
 * `lengths [batch]`, a 16-word uniform block and a u32 tensor-offset table.
 */
import type { ModelManifest } from "./weights.ts";

export interface PackedRows {
  rows: Uint32Array;
  lengths: Uint32Array;
  maxTokens: number;
}

/** Packs feature rows of several sequences into one `[batch, maxTokens, slots]` buffer. */
export function packRows(
  batch: ArrayLike<ArrayLike<ArrayLike<number>>>,
  slots: number,
  paddingId: number,
): PackedRows {
  let maxTokens = 1;
  for (let s = 0; s < batch.length; s++) maxTokens = Math.max(maxTokens, batch[s]!.length);
  const rows = new Uint32Array(batch.length * maxTokens * slots).fill(paddingId);
  const lengths = new Uint32Array(batch.length);
  for (let s = 0; s < batch.length; s++) {
    const seq = batch[s]!;
    lengths[s] = seq.length;
    for (let t = 0; t < seq.length; t++) {
      const row = seq[t]!;
      const base = (s * maxTokens + t) * slots;
      for (let i = 0; i < Math.min(slots, row.length); i++) rows[base + i] = row[i]!;
    }
  }
  return { rows, lengths, maxTokens };
}

/** Offsets of the named tensors into the flat decoded weights, in the given order. */
export function tensorOffsets(manifest: ModelManifest, names: string[]): Uint32Array {
  return Uint32Array.from(names, (name) => {
    const entry = manifest.tensors.find((t) => t.name === name);
    if (!entry) throw new Error(`unknown tensor ${name}`);
    return entry.offset;
  });
}

export interface TaggerParams {
  batch: number;
  maxTokens: number;
  slots: number;
  padding: number;
  embed: number;
  hidden: number;
  head: number;
  tags: number;
  pooled: number;
  layers: number;
  taps: number;
}

/** The 16-word `Params` uniform shared by scan_tagger.wgsl and conv_tagger.wgsl. */
export function taggerParams(p: TaggerParams): Uint32Array {
  const out = new Uint32Array(16);
  out.set([
    p.batch,
    p.maxTokens,
    p.slots,
    p.padding,
    p.embed,
    p.hidden,
    p.head,
    p.tags,
    p.pooled,
    p.layers,
    p.taps,
  ]);
  return out;
}

/** Workgroup grid for `n` items, wrapped at 32768 columns (dispatch limit is 65535 per axis). */
export function grid(n: number): [number, number] {
  const GRID_X = 32768;
  return [Math.min(Math.max(n, 1), GRID_X), Math.max(1, Math.ceil(n / GRID_X))];
}

/** Splits the flat kernel output into per-sequence tag logits and pooled logits. */
export function unpackLogits(
  out: Float32Array,
  lengths: Uint32Array,
  maxTokens: number,
  tags: number,
  pooled: number,
): { tags: Float32Array[]; pooled?: Float32Array[] } {
  const batch = lengths.length;
  const perSeq: Float32Array[] = [];
  for (let s = 0; s < batch; s++) {
    perSeq.push(out.slice(s * maxTokens * tags, s * maxTokens * tags + lengths[s]! * tags));
  }
  if (!(pooled > 0)) return { tags: perSeq };
  const base = batch * maxTokens * tags;
  const pooledOut: Float32Array[] = [];
  for (let s = 0; s < batch; s++)
    pooledOut.push(out.slice(base + s * pooled, base + (s + 1) * pooled));
  return { tags: perSeq, pooled: pooledOut };
}
