/**
 * gpu-view: Natural language to table view specs: filter, sort, group, aggregate, limit, chart
 *
 * Public API. Keep this file small: the CPU tokenizer/featurizer lives in ./features.ts,
 * the reference forward pass in ./cpu.ts, the WebGPU forward pass in ./gpu.ts, and the
 * decoder in ./decode.ts. Both forward passes must produce identical logits.
 */
import { type Backend, hasWebGPU } from "@gpu-utils/runtime";
import { forwardCpu } from "./cpu.ts";
import { type GpuViewResult, decode } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpu } from "./gpu.ts";
import { MODEL } from "./model.ts";

export type { GpuViewResult } from "./decode.ts";

export interface GpuViewOptions {
  /** "auto" uses WebGPU for large inputs when available and the CPU reference otherwise. */
  backend?: Backend;
}

/** Inputs shorter than this run on the CPU under "auto": GPU readback latency dominates. */
const GPU_MIN_TOKENS = 256;

export async function parse(
  text: string,
  options: GpuViewOptions = {},
): Promise<GpuViewResult> {
  const features = featurize(text);
  const backend = options.backend ?? "auto";
  const useGpu =
    backend === "webgpu" ||
    (backend === "auto" && hasWebGPU() && features.tokens.length >= GPU_MIN_TOKENS);
  const logits = useGpu ? await forwardGpu(MODEL, features) : forwardCpu(MODEL, features);
  return decode(MODEL, features, logits);
}
