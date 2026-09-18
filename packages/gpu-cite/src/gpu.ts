import {
  createProgram,
  type FeatureRows,
  type Program,
  runScanTagger,
  scanTaggerEntries,
  scanTaggerShader,
} from "@gpu-utils/runtime";
import { type Logits, splitLogits } from "./cpu.ts";
import type { Model } from "./model.ts";

let programPromise: Promise<Program> | undefined;

function program(model: Model): Promise<Program> {
  programPromise ??= createProgram(
    scanTaggerShader,
    scanTaggerEntries(model.manifest.scanLayers ?? 1),
  );
  return programPromise;
}

/**
 * WebGPU forward pass over a batch of references in one command buffer, on the canonical
 * scan-family kernel. Produces the same logits as forwardCpu for every sequence (checked by
 * training/tests/test_wgsl.py on lavapipe).
 */
export async function forwardGpuBatch(model: Model, batch: FeatureRows[]): Promise<Logits[]> {
  if (batch.length === 0) return [];
  const out = await runScanTagger(
    await program(model),
    model,
    batch.map((f) => f.rows),
  );
  return batch.map((f, i) => {
    const tags = out.tags[i]!;
    const pooled = out.pooled?.[i];
    return splitLogits(model, pooled ? { tags, pooled } : { tags }, f.rows.length);
  });
}

/** Single-reference convenience wrapper around forwardGpuBatch. */
export async function forwardGpu(model: Model, features: FeatureRows): Promise<Logits> {
  const [logits] = await forwardGpuBatch(model, [features]);
  return logits!;
}
