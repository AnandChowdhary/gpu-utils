import { decodeInt6, type TaggerManifest, type TaggerModel } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

/**
 * The shared scan-family manifest (gpu_utils_training.export.export_package) plus the
 * two label lists gpu-paste needs to split the family's outputs: `labels` names the BIO
 * span head (`tags`), `kinds` names the pooled kind head (`pooled`).
 */
export interface PasteManifest extends TaggerManifest {
  /** BIO labels for the span head: "O", "B-person", "I-person", ... */
  labels: string[];
  /** Learned kinds, in pooled-head order. */
  kinds: string[];
  spanKinds: string[];
}

export interface Model extends TaggerModel {
  manifest: PasteManifest;
}

/** Trained weights, decoded once at import time. Written by training/gpu_paste/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as PasteManifest,
  weights: decodeInt6(encoded, (manifest as unknown as PasteManifest).tensors),
};
