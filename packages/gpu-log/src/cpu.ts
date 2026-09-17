import type { FeatureRows } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass in plain TypeScript. Returns [tokens, labels] logits.
 * This is the source of truth the WGSL kernels are checked against, and the
 * fallback for small inputs and for environments without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  const n = features.tokens.length;
  const k = model.manifest.labels.length;
  const logits = new Float32Array(n * k);
  // TODO: embed → sequence mixing → head, mirroring training/gpu_log/model.py.
  return logits;
}
