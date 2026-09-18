import {
  type BatchOutput,
  convTaggerEntries,
  convTaggerShader,
  createProgram,
  type Program,
  runConvTagger,
} from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

let programPromise: Promise<Program> | undefined;

function program(model: Model): Promise<Program> {
  programPromise ??= createProgram(
    convTaggerShader,
    convTaggerEntries((model.manifest.dilations ?? []).length),
  );
  return programPromise;
}

/**
 * WebGPU forward pass over a batch of lines on the canonical conv_tagger kernel: one dispatch
 * of `embed → block0..4 → head → pool` over `[batch, maxTokens]` padded rows. Produces the
 * same per-line tag and kind logits as forwardCpu.
 */
export async function forwardGpuBatch(model: Model, batch: number[][][]): Promise<BatchOutput> {
  return runConvTagger(await program(model), model, batch);
}
