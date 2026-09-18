import { convTaggerForward, type FeatureRows } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass in plain TypeScript: the runtime's conv-family implementation
 * (embed → projection → six residual dilated conv blocks → per-token head), mirroring
 * training/gpu_email/model.py. Returns one row of `manifest.tags` logits per token: the
 * line-kind columns first, then the BIO contact-field columns (see decode.ts). This is the
 * source of truth the WebGPU path is checked against and the fallback without WebGPU.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
  return convTaggerForward(model, features.rows).tags;
}
