import { decodeInt6, type ModelManifest, tensor } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface Model {
  manifest: ModelManifest;
  /** Flat float32 weights; use `tensor(weights, manifest, name)` for a named view. */
  weights: Float32Array;
  /** Embedding width D. */
  hidden: number;
  /** Head width H. */
  head: number;
  /** Output columns: role labels + 1 boundary logit. */
  out: number;
  width: number;
}

const m = manifest as ModelManifest;

/** Trained weights, decoded once at import time. Written by training/gpu_tailwind/export.py. */
export const MODEL: Model = {
  manifest: m,
  weights: decodeInt6(encoded, m.tensors),
  hidden: Number(m.hidden ?? 24),
  head: Number(m.head ?? 32),
  out: Number(m.out ?? m.labels.length + 1),
  width: Number(m.width ?? 7),
};

export function weight(model: Model, name: string): Float32Array {
  return tensor(model.weights, model.manifest, name);
}
