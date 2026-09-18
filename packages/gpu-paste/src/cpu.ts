import { scanTaggerForward } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

export interface Logits {
  /** [tokens, labels] span logits, row-major (the family's per-token `tags`). */
  span: Float32Array;
  /** [kinds] logits from the mean-pooled kind head (the family's `pooled` output). */
  kind: Float32Array;
}

/**
 * Reference forward pass: the runtime's canonical scan family (`scanTaggerForward`,
 * mirroring gpu_utils_training.models.ScanTagger). gpu-paste is a plain
 * `ScanTagger(feature_rows, hidden, tags, pooled_out=kinds)`: the BIO span head is the
 * family's `tags` output and the kind head its `pooled` output. This is the source of
 * truth the WebGPU path (runtime `runScanTagger`) is checked against.
 */
export function forwardCpu(model: Model, rows: number[][]): Logits {
  const out = scanTaggerForward(model, rows);
  return { span: out.tags, kind: out.pooled ?? new Float32Array(model.manifest.kinds.length) };
}
