import { type FeatureRows, scanTaggerForward } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass: the runtime's scan-family implementation (mirrors
 * gpu_utils_training.models.ScanTagger). Returns `[tokens, tags]` logits where the first
 * `labels.length` columns are role logits and the last column is the segment-boundary logit.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  return scanTaggerForward(model, features.rows).tags;
}
