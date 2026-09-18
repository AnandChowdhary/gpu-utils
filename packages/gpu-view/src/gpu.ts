import { createProgram, type Program } from "@gpu-utils/runtime";
import type { ViewFeatures } from "./features.ts";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

let programPromise: Promise<Program> | undefined;

const OFFSET_NAMES = [
  "embedding",
  "encoder_bias",
  "convolution",
  "gate_weight",
  "gate_bias",
  "candidate_weight",
  "candidate_bias",
  "combine_weight",
  "combine_bias",
  "global_weight",
  "global_bias",
  "head_gate_weight",
  "head_gate_bias",
  "head_hidden_weight",
  "head_hidden_bias",
  "output_weight",
  "output_bias",
];

/** Packs a batch of feature rows for the kernel. Exported for the WGSL parity harness. */
export function packBatch(model: Model, batch: ViewFeatures[]) {
  const slots = model.manifest.slots;
  const padding = model.manifest.featureRows;
  const maxTokens = Math.max(1, ...batch.map((f) => f.tokens.length));
  const rows = new Uint32Array(batch.length * maxTokens * slots).fill(padding);
  const lengths = new Uint32Array(batch.length);
  batch.forEach((f, s) => {
    lengths[s] = f.tokens.length;
    f.rows.forEach((row, t) => {
      rows.set(row.slice(0, slots), (s * maxTokens + t) * slots);
    });
  });
  const offsets = new Uint32Array(20);
  OFFSET_NAMES.forEach((name, i) => {
    offsets[i] = model.manifest.tensors.find((t) => t.name === name)!.offset;
  });
  const outs = model.manifest.labels.length + 1;
  const params = new Uint32Array([
    batch.length,
    maxTokens,
    slots,
    outs,
    model.manifest.hidden,
    model.manifest.headGate,
    padding,
    0,
  ]);
  return { rows, lengths, offsets, params, maxTokens, outs };
}

/** WebGPU forward pass over a batch: one workgroup per phrase. Matches forwardCpu per phrase. */
export async function forwardGpuBatch(
  model: Model,
  batch: ViewFeatures[],
): Promise<Float32Array[]> {
  programPromise ??= createProgram(shader, ["tag"]);
  const program = await programPromise;
  const { rows, lengths, offsets, params, maxTokens, outs } = packBatch(model, batch);
  const n = batch.length;
  const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const paramsBuf = program.buffer("params", params, uniform);
  const offsetsBuf = program.buffer("offsets", offsets, uniform);
  const weights = program.buffer("weights", model.weights);
  const rowsBuf = program.buffer("rows", rows);
  const lengthsBuf = program.buffer("lengths", lengths);
  const scratch = program.buffer(
    "scratch",
    new Float32Array(Math.max(1, 4 * n * maxTokens * model.manifest.hidden)),
  );
  const logits = program.buffer(
    "logits",
    new Float32Array(Math.max(1, n * maxTokens * outs)),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const out = await program.run(
    [paramsBuf, offsetsBuf, weights, rowsBuf, lengthsBuf, scratch, logits],
    [{ entry: "tag", workgroups: [n] }],
    logits,
  );
  return batch.map((f, s) =>
    out.slice(s * maxTokens * outs, s * maxTokens * outs + f.tokens.length * outs),
  );
}
