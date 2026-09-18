import { decodeInt6, type ModelManifest, tensor } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface EmailManifest extends ModelManifest {
  /** Line kinds (the `labels` of the runtime manifest). */
  labels: string[];
  /** BIO labels of the contact-field head. */
  fields: string[];
  /** Embedding / trunk width. */
  hidden: number;
  /** Width of the shared head hidden layer. */
  head: number;
  dilations: number[];
  slots: number;
  rows: number;
}

export interface Model {
  manifest: EmailManifest;
  /** Flat float32 weights; use `tensor(weights, manifest, name)` for a named view. */
  weights: Float32Array;
}

/** Trained weights, decoded once at import time. Written by training/gpu_email/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as EmailManifest,
  weights: decodeInt6(encoded, (manifest as unknown as EmailManifest).tensors),
};

/** Number of logits per token: line kinds followed by BIO labels. */
export function logitWidth(model: Model): number {
  return model.manifest.labels.length + model.manifest.fields.length;
}

export function weight(model: Model, name: string): Float32Array {
  return tensor(model.weights, model.manifest, name);
}
