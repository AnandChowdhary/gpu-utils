import { decodeInt6, type TaggerManifest, type TaggerModel } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface LogManifest extends TaggerManifest {
  /** Names of the pooled-head classes: entry, continuation, frame. */
  kinds: string[];
}

export interface Model extends TaggerModel {
  manifest: LogManifest;
}

/** Trained weights, decoded once at import time. Written by training/gpu_log/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as LogManifest,
  weights: decodeInt6(encoded, (manifest as unknown as LogManifest).tensors),
};
