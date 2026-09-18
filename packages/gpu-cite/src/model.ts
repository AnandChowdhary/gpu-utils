import { decodeInt6, type TaggerManifest, type TaggerModel } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

/** The scan-family manifest plus the label inventories the decoder needs. */
export interface CiteManifest extends TaggerManifest {
  /** Field roles; `labels` is "O" + B-role + I-role over these. */
  roles: string[];
  /** Document types emitted by the pooled head. */
  types: string[];
  /** Name-part labels carried by the tag columns after the BIO labels. */
  nameparts: string[];
}

export interface Model extends TaggerModel {
  manifest: CiteManifest;
}

/** Trained weights, decoded once at import time. Written by training/gpu_cite/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as CiteManifest,
  weights: decodeInt6(encoded, (manifest as unknown as CiteManifest).tensors),
};
