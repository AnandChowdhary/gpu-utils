import {
  convTaggerEntries,
  convTaggerShader,
  createProgram,
  type FeatureRows,
  type Program,
  runConvTagger,
} from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

let programPromise: Promise<Program> | undefined;

function program(model: Model): Promise<Program> {
  programPromise ??= createProgram(
    convTaggerShader,
    convTaggerEntries(model.manifest.dilations?.length ?? 0),
  );
  return programPromise;
}

/** WebGPU forward pass over a batch on the canonical conv kernel (one workgroup per token). */
export async function forwardGpuBatch(model: Model, batch: FeatureRows[]): Promise<Float32Array[]> {
  const out = await runConvTagger(
    await program(model),
    model,
    batch.map((f) => f.rows),
  );
  return out.tags;
}

/** WebGPU forward pass. Produces the same logits as forwardCpu (checked by training/tests/test_wgsl.py). */
export async function forwardGpu(model: Model, features: FeatureRows): Promise<Float32Array> {
  if (features.rows.length === 0) return new Float32Array(0);
  const [logits] = await forwardGpuBatch(model, [features]);
  return logits!;
}
