import { createProgram, type Pass, type Program } from "@gpu-utils/runtime";
import type { Batch } from "./cpu.ts";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

const ENTRIES = ["embed", "block0", "block1", "block2", "block3", "block4", "heads"];
const GRID_X = 32768;

let programPromise: Promise<Program> | undefined;

/** Tensor offsets in the order shader.wgsl reads them from `offs`. */
export function weightOffsets(model: Model): Uint32Array {
  const names = ["embed", "proj_w", "proj_b"];
  for (let i = 0; i < model.blocks.length; i++)
    names.push(`block${i}_w1`, `block${i}_b1`, `block${i}_w2`, `block${i}_b2`);
  names.push("head_h_w", "head_h_b", "head_tag_w", "head_tag_b", "head_kind_w", "head_kind_b");
  return Uint32Array.from(names, (n) => model.offsets[n]!);
}

/** Workgroup grid for n tokens: one workgroup per token, wrapped at GRID_X columns. */
export function grid(n: number): [number, number] {
  return [Math.min(n, GRID_X), Math.ceil(n / GRID_X)];
}

/**
 * WebGPU forward pass for one packed batch of lines. Produces the same [n, tags + kinds]
 * logits as forwardCpu; all seven passes run in one command buffer with a single readback.
 */
export async function forwardGpu(model: Model, batch: Batch): Promise<Float32Array> {
  if (
    model.hidden !== 64 ||
    model.embedDim !== 32 ||
    model.featureCount !== 9 ||
    model.tags !== 23 ||
    model.kinds !== 3 ||
    model.blocks.length !== 5
  ) {
    throw new Error("shader.wgsl is compiled for H=64, E=32, F=9, T=23, K=3, 5 blocks");
  }
  programPromise ??= createProgram(shader, ENTRIES);
  const program = await programPromise;
  const n = batch.n;
  const W = model.tags + model.kinds;
  if (n === 0) return new Float32Array(0);

  const params = program.buffer(
    "params",
    new Uint32Array([n, 0, 0, 0]),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const offs = program.buffer("offs", weightOffsets(model));
  const features = program.buffer("features", batch.features);
  const lineId = program.buffer("line_id", batch.lineId);
  const weights = program.buffer("weights", model.weights);
  const stateA = program.buffer("state_a", new Float32Array(n * model.hidden));
  const stateB = program.buffer("state_b", new Float32Array(n * model.hidden));
  const logits = program.buffer(
    "logits",
    new Float32Array(n * W),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const workgroups = grid(n);
  const passes: Pass[] = ENTRIES.map((entry) => ({ entry, workgroups }));
  const out = await program.run(
    [params, offs, features, lineId, weights, stateA, stateB, logits],
    passes,
    logits,
  );
  return out.subarray(0, n * W);
}
