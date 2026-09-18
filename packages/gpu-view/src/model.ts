import { decodeInt6, type ModelManifest } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface ViewManifest extends ModelManifest {
  hidden: number;
  headGate: number;
  conv: number;
  featureRows: number;
  slots: number;
}

export interface Model {
  manifest: ViewManifest;
  /** Flat float32 weights; use `tensor(weights, manifest, name)` for a named view. */
  weights: Float32Array;
}

/** Trained weights, decoded once at import time. Written by training/gpu_view/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as ViewManifest,
  weights: decodeInt6(encoded, (manifest as unknown as ViewManifest).tensors),
};
