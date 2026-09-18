import {
  type BioSpan,
  bioStartMask,
  bioToSpans,
  bioTransitions,
  type FeatureRows,
  viterbi,
} from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

export interface __PASCAL__Span {
  label: string;
  text: string;
  /** UTF-16 offsets, half-open. */
  start: number;
  end: number;
}

export interface __PASCAL__Result {
  /** One label per token, aligned with `tokens`. */
  labels: string[];
  spans: __PASCAL__Span[];
  tokens: { text: string; start: number; end: number }[];
}

/**
 * Turns logits into the typed public result: constrained Viterbi over BIO tags, then a
 * deterministic compiler from spans to the output type. All semantics live here.
 */
export function decode(
  model: Model,
  features: FeatureRows,
  logits: Float32Array,
  text: string,
): __PASCAL__Result {
  const labels = model.manifest.labels;
  const k = labels.length;
  const n = features.tokens.length;
  const emissions = Float32Array.from(logits);
  const start = bioStartMask(labels);
  for (let j = 0; j < k && n > 0; j++) emissions[j]! += start[j]!;
  const ids = viterbi(emissions, n, k, bioTransitions(labels));
  const spans = bioToSpans(ids, labels, features.tokens).map((s: BioSpan) => ({
    label: s.label,
    text: text.slice(s.start, s.end),
    start: s.start,
    end: s.end,
  }));
  return {
    labels: Array.from(ids, (i) => labels[i] ?? "O"),
    spans,
    tokens: features.tokens.map(({ text, start, end }) => ({ text, start, end })),
  };
}
