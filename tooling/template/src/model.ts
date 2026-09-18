import { decodeInt6, type TaggerManifest, type TaggerModel } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export type Model = TaggerModel;

/** Trained weights, decoded once at import time. Written by training/__SNAKE__/export.py. */
export const MODEL: Model = {
  manifest: manifest as unknown as TaggerManifest,
  weights: decodeInt6(encoded, (manifest as unknown as TaggerManifest).tensors),
};
