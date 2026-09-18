import { type FeatureRows, scanTaggerForward } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass in plain TypeScript. Returns [tokens, labels] logits.
 * The model is a shared family (see training/__SNAKE__/model.py), so this is just the
 * runtime's reference implementation; it is the source of truth the WebGPU path is
 * checked against and the fallback for small inputs and environments without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  return scanTaggerForward(model, features.rows).tags;
}
