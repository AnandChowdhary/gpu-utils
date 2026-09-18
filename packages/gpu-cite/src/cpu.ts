import { type FeatureRows, scanTaggerForward, type TaggerOutput } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/** Model outputs for one reference: [n, K] BIO tag logits, [n, P] name-part logits, [T] type logits. */
export interface Logits {
  n: number;
  tags: Float32Array;
  parts: Float32Array;
  type: Float32Array;
}

/**
 * Splits the family output into the decoder's view. The model is a shared `ScanTagger`
 * (see training/gpu_cite/model.py): per token the first K tag columns are the BIO
 * emissions and the remaining P columns the name-part head; the pooled head is the
 * document type.
 */
export function splitLogits(model: Model, out: TaggerOutput, n: number): Logits {
  const m = model.manifest;
  const K = m.labels.length;
  const P = m.nameparts.length;
  const width = m.tags;
  const tags = new Float32Array(n * K);
  const parts = new Float32Array(n * P);
  for (let i = 0; i < n; i++) {
    tags.set(out.tags.subarray(i * width, i * width + K), i * K);
    parts.set(out.tags.subarray(i * width + K, i * width + K + P), i * P);
  }
  return { n, tags, parts, type: out.pooled ?? new Float32Array(m.types.length) };
}

/**
 * Reference forward pass: the runtime's scan-family CPU implementation. It is the source
 * of truth the WebGPU path is checked against and the fallback for small inputs and
 * environments without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Logits {
  return splitLogits(model, scanTaggerForward(model, features.rows), features.rows.length);
}
