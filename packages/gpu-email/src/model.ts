import { decodeInt6, type TaggerManifest, type TaggerModel } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface EmailManifest extends TaggerManifest {
  /** Line kinds: the first `labels.length` tag columns. */
  labels: string[];
  /** BIO labels of the contact-field head: the remaining tag columns. */
  fields: string[];
}

export interface Model extends TaggerModel {
  manifest: EmailManifest;
}

/** Trained weights, decoded once at import time. Written by training/gpu_email/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as EmailManifest,
  weights: decodeInt6(encoded, (manifest as unknown as EmailManifest).tensors),
};

/** Number of logits per token: line kinds followed by BIO labels. */
export function logitWidth(model: Model): number {
  return model.manifest.tags;
}
