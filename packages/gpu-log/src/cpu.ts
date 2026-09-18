import { convTaggerForward, type TaggerOutput } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass for one line: the runtime's conv-family implementation
 * (mirrors gpu_utils_training.models.ConvTagger). Returns `[n, tags]` logits plus the
 * `[kinds]` pooled logits. The WebGPU path (gpu.ts) must match it to 1e-4.
 */
export function forwardCpu(model: Model, rows: number[][]): TaggerOutput {
  return convTaggerForward(model, rows);
}
