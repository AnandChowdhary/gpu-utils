import { decodeInt6, type ModelManifest, tensor } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface PasteManifest extends ModelManifest {
  dim: number;
  mix: number;
  kindHidden: number;
  featureCount: number;
  rows: number;
  /** BIO labels for the span head: "O", "B-person", "I-person", ... */
  labels: string[];
  /** Learned kinds, in kind-head order. */
  kinds: string[];
  spanKinds: string[];
}

export interface Model {
  manifest: PasteManifest;
  /** Flat float32 weights; `tensor(weights, manifest, name)` gives a named view. */
  weights: Float32Array;
}

/** Trained weights, decoded once at import time. Written by training/gpu_paste/export.py. */
export const MODEL: Model = {
  manifest: manifest as PasteManifest,
  weights: decodeInt6(encoded, (manifest as PasteManifest).tensors),
};

export const view = (model: Model, name: string): Float32Array =>
  tensor(model.weights, model.manifest, name);
