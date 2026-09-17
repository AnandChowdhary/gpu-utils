import { decodeInt6, type ModelManifest, tensor } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface LogManifest extends ModelManifest {
  hidden: number;
  embedDim: number;
  blocks: number;
  featureCount: number;
  embedRows: number;
  kinds: string[];
}

export interface BlockWeights {
  dilation: number;
  /** [3, H, H] as (tap, in, out). */
  w1: Float32Array;
  b1: Float32Array;
  /** [H, H] as (in, out). */
  w2: Float32Array;
  b2: Float32Array;
}

export interface Model {
  manifest: LogManifest;
  /** Flat float32 weights, decoded once; the views below index into it. */
  weights: Float32Array;
  hidden: number;
  embedDim: number;
  featureCount: number;
  tags: number;
  kinds: number;
  embed: Float32Array;
  projW: Float32Array;
  projB: Float32Array;
  blocks: BlockWeights[];
  headHW: Float32Array;
  headHB: Float32Array;
  headTagW: Float32Array;
  headTagB: Float32Array;
  headKindW: Float32Array;
  headKindB: Float32Array;
  /** Byte offsets (in floats) of every tensor, for the GPU path. */
  offsets: Record<string, number>;
}

export function loadModel(m: LogManifest, weightsText: string): Model {
  const weights = decodeInt6(weightsText, m.tensors);
  const view = (name: string) => tensor(weights, m, name);
  const offsets: Record<string, number> = {};
  for (const t of m.tensors) offsets[t.name] = t.offset;
  const blocks: BlockWeights[] = [];
  for (let i = 0; i < m.blocks; i++) {
    blocks.push({
      dilation: 1 << i,
      w1: view(`block${i}_w1`),
      b1: view(`block${i}_b1`),
      w2: view(`block${i}_w2`),
      b2: view(`block${i}_b2`),
    });
  }
  return {
    manifest: m,
    weights,
    hidden: m.hidden,
    embedDim: m.embedDim,
    featureCount: m.featureCount,
    tags: m.labels.length,
    kinds: m.kinds.length,
    embed: view("embed"),
    projW: view("proj_w"),
    projB: view("proj_b"),
    blocks,
    headHW: view("head_h_w"),
    headHB: view("head_h_b"),
    headTagW: view("head_tag_w"),
    headTagB: view("head_tag_b"),
    headKindW: view("head_kind_w"),
    headKindB: view("head_kind_b"),
    offsets,
  };
}

/** Trained weights, decoded once at import time. Written by training/gpu_log/export.py. */
export const MODEL: Model = loadModel(manifest as LogManifest, encoded);
