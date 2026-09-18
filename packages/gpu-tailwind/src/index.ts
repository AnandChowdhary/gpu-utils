/**
 * gpu-tailwind: Natural language to Tailwind CSS utility classes.
 *
 * "card with rounded corners, subtle shadow, blue on hover, hidden on mobile"
 *   -> rounded-lg bg-white p-4 shadow-sm hover:bg-blue-500 max-sm:hidden
 *
 * A ~45K-parameter tagger labels each token PROPERTY / VALUE / VARIANT / SEP / NEG and
 * scores segment boundaries; a deterministic compiler (src/compile.ts) turns the tags into
 * classes through a table compiled from the Tailwind v4 default theme. The model never
 * generates text, and every emitted class is validated against that vocabulary.
 */
import { type Backend, hasWebGPU } from "@gpu-utils/runtime";
import { forwardCpu } from "./cpu.ts";
import { decode, type TailwindResult } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpu } from "./gpu.ts";
import { MODEL } from "./model.ts";

export type { Diagnostic, Group, TailwindResult } from "./decode.ts";
export { isValidClass } from "./compile.ts";

export interface TailwindOptions {
  /** "auto" uses WebGPU for large inputs when available and the CPU reference otherwise. */
  backend?: Backend;
}

/** Inputs shorter than this run on the CPU under "auto": GPU readback latency dominates. */
const GPU_MIN_TOKENS = 256;

export async function parse(text: string, options: TailwindOptions = {}): Promise<TailwindResult> {
  const features = featurize(text);
  const backend = options.backend ?? "auto";
  const useGpu =
    backend === "webgpu" ||
    (backend === "auto" && hasWebGPU() && features.tokens.length >= GPU_MIN_TOKENS);
  const logits = useGpu ? await forwardGpu(MODEL, features) : forwardCpu(MODEL, features);
  return decode(MODEL, features, logits);
}

/** Batch helper: one call per phrase, sharing the loaded model. */
export async function parseMany(texts: string[], options: TailwindOptions = {}): Promise<TailwindResult[]> {
  const out: TailwindResult[] = [];
  for (const t of texts) out.push(await parse(t, options));
  return out;
}
