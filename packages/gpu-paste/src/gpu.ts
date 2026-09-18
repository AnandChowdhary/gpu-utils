import {
  createProgram,
  type Program,
  runScanTagger,
  scanTaggerEntries,
  scanTaggerShader,
} from "@gpu-utils/runtime";
import { forwardCpu, type Logits } from "./cpu.ts";
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
 * WebGPU forward pass over a batch of windows in one dispatch, on the runtime's canonical
 * scan kernel (`runScanTagger`). Produces the same logits as forwardCpu per window; parity
 * is checked on lavapipe by training/tests/test_wgsl.py.
 */
export async function forwardGpuBatch(model: Model, batch: number[][][]): Promise<Logits[]> {
  const out = await runScanTagger(await program(model), model, batch);
  return batch.map((rows, i) =>
    rows.length === 0
      ? forwardCpu(model, rows)
      : { span: out.tags[i]!, kind: out.pooled?.[i] ?? forwardCpu(model, []).kind },
  );
}

/** WebGPU forward pass for one window. */
export async function forwardGpu(model: Model, rows: number[][]): Promise<Logits> {
  if (rows.length === 0) return forwardCpu(model, rows);
  const [logits] = await forwardGpuBatch(model, [rows]);
  return logits!;
}
