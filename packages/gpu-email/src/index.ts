/**
 * gpu-email: Split plain-text emails into reply, quoted history, and signature, and
 * extract contact details.
 *
 * Public API. The CPU tokenizer/featurizer lives in ./features.ts, the reference forward
 * pass in ./cpu.ts, the WebGPU forward pass in ./gpu.ts, and the decoder in ./decode.ts.
 * Both forward passes must produce identical logits.
 */
import { type Backend, hasWebGPU } from "@gpu-utils/runtime";
import { forwardCpu } from "./cpu.ts";
import { decode, type EmailParseResult } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpu } from "./gpu.ts";
import { MODEL } from "./model.ts";

export type {
  Contact,
  Diagnostics,
  EmailParseResult,
  Segment,
  SegmentKind,
} from "./decode.ts";

export interface EmailParseOptions {
  /** "auto" uses WebGPU for large inputs when available and the CPU reference otherwise. */
  backend?: Backend;
}

export class EmailInputError extends Error {
  override name = "EmailInputError";
}

/** Inputs shorter than this run on the CPU under "auto": GPU readback latency dominates. */
const GPU_MIN_TOKENS = 512;

/**
 * Parses one plain-text email body (after MIME decoding, no HTML) into labelled
 * segments, the new reply text, and the author's contact details when a signature
 * block is present.
 */
export async function parse(
  text: string,
  options: EmailParseOptions = {},
): Promise<EmailParseResult> {
  if (typeof text !== "string") throw new EmailInputError("parse() expects a string");
  const t0 = performance.now();
  const features = featurize(text);
  const backend = options.backend ?? "auto";
  const useGpu =
    backend === "webgpu" ||
    (backend === "auto" && hasWebGPU() && features.tokens.length >= GPU_MIN_TOKENS);
  let logits: Float32Array;
  let used: "cpu" | "webgpu" = "cpu";
  if (useGpu) {
    try {
      logits = await forwardGpu(MODEL, features);
      used = "webgpu";
    } catch (err) {
      if (backend === "webgpu") throw err;
      logits = forwardCpu(MODEL, features);
    }
  } else {
    logits = forwardCpu(MODEL, features);
  }
  const result = decode(MODEL, features, logits, text);
  return {
    ...result,
    diagnostics: {
      backend: used,
      tokens: features.tokens.length,
      lines: features.lines.length,
      ms: performance.now() - t0,
    },
  };
}

/** Parses several emails; each runs independently (the GPU program is shared). */
export async function parseMany(
  texts: string[],
  options: EmailParseOptions = {},
): Promise<EmailParseResult[]> {
  const out: EmailParseResult[] = [];
  for (const t of texts) out.push(await parse(t, options));
  return out;
}
