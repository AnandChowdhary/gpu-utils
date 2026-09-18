import {
  createProgram,
  type FeatureRows,
  type Program,
  runScanTagger,
  scanTaggerEntries,
  scanTaggerShader,
} from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

let programPromise: Promise<Program> | undefined;

function program(model: Model): Promise<Program> {
  programPromise ??= createProgram(
    scanTaggerShader,
    scanTaggerEntries(model.manifest.scanLayers ?? 1),
  );
  return programPromise;
}

/** WebGPU forward pass over a batch (one workgroup per sequence in the scan passes). */
export async function forwardGpuBatch(model: Model, batch: FeatureRows[]): Promise<Float32Array[]> {
  const out = await runScanTagger(
    await program(model),
    model,
    batch.map((f) => f.rows),
  );
  return out.tags;
}

/** WebGPU forward pass. Produces the same logits as forwardCpu (checked by the WGSL harness). */
export async function forwardGpu(model: Model, features: FeatureRows): Promise<Float32Array> {
  const [logits] = await forwardGpuBatch(model, [features]);
  return logits!;
}
