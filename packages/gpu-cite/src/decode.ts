import { argmax, type FeatureRows } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

export interface GpuCiteResult {
  /** One label per token, aligned with `tokens`. */
  labels: string[];
  tokens: { text: string; start: number; end: number }[];
}

/** Turns logits into the typed public result. Replace argmax with viterbi once a CRF exists. */
export function decode(
  model: Model,
  features: FeatureRows,
  logits: Float32Array,
): GpuCiteResult {
  const k = model.manifest.labels.length;
  const ids = argmax(logits, features.tokens.length, k);
  return {
    labels: Array.from(ids, (i) => model.manifest.labels[i] ?? "O"),
    tokens: features.tokens.map(({ text, start, end }) => ({ text, start, end })),
  };
}
