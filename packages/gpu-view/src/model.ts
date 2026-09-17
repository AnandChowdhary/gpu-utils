import { decodeInt6, type ModelManifest } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface Model {
  manifest: ModelManifest;
  /** Flat float32 weights; use `tensor(weights, manifest, name)` for a named view. */
  weights: Float32Array;
}

/** Trained weights, decoded once at import time. Written by training/gpu_view/export.py. */
export const MODEL: Model = {
  manifest: manifest as ModelManifest,
  weights: decodeInt6(encoded, (manifest as ModelManifest).tensors),
};
