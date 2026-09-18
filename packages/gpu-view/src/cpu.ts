import { type FeatureRows, scanTaggerForward } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass: the runtime's scan-family forward. Returns `[tokens, tags]`
 * logits where the first 14 columns are the roles and the last is the clause-boundary
 * logit. The WebGPU path (gpu.ts) runs the canonical kernel and must match this to 1e-4.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  return scanTaggerForward(model, features.rows).tags;
}
