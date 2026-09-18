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

/** WebGPU forward pass over a batch on the canonical scan-family kernel (one workgroup per phrase in the scan passes). */
export async function forwardGpuBatch(model: Model, batch: FeatureRows[]): Promise<Float32Array[]> {
  if (batch.length === 0) return [];
  const out = await runScanTagger(
    await program(model),
    model,
    batch.map((f) => f.rows),
  );
  return out.tags;
}
