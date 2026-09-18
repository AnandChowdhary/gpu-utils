import { createProgram, type FeatureRows, type Program } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

export const ENTRIES = ["embed", "gates", "scan_local", "scan_fixup", "pool", "head"];
export const TENSOR_ORDER = [
  "emb",
  "wa_f",
  "ba_f",
  "wu_f",
  "bu_f",
  "wa_b",
  "ba_b",
  "wu_b",
  "bu_b",
  "conv",
  "bc",
  "w1",
  "wg",
  "b1",
  "w2",
  "b2",
];
const CHUNK = 256;

let programPromise: Promise<Program> | undefined;

/** Uniform block layout shared with training/tests/test_wgsl.py. */
export function packParams(model: Model, n: number): Uint32Array {
  const params = new Uint32Array(24);
  params[0] = n;
  params[1] = Math.ceil(n / CHUNK);
  params[2] = model.out;
  params[3] = model.width;
  TENSOR_ORDER.forEach((name, i) => {
    const t = model.manifest.tensors.find((e) => e.name === name);
    if (!t) throw new Error(`missing tensor ${name}`);
    params[4 + i] = t.offset;
  });
  return params;
}

/** Workgroup counts per pass for n tokens (shared with the Python harness). */
export function dispatch(
  model: Model,
  n: number,
): { entry: string; workgroups: [number, number?, number?] }[] {
  const D = model.hidden;
  return [
    { entry: "embed", workgroups: [Math.ceil((n * D) / 64)] },
    { entry: "gates", workgroups: [Math.ceil((2 * n * D) / 64)] },
    { entry: "scan_local", workgroups: [Math.ceil(n / CHUNK), 2 * D] },
    { entry: "scan_fixup", workgroups: [Math.ceil((2 * n * D) / 64)] },
    { entry: "pool", workgroups: [1] },
    { entry: "head", workgroups: [Math.ceil(n / 64)] },
  ];
}

/** WebGPU forward pass. Produces the same logits as forwardCpu (checked by training/tests/test_wgsl.py). */
export async function forwardGpu(model: Model, features: FeatureRows): Promise<Float32Array> {
  programPromise ??= createProgram(shader, ENTRIES);
  const program = await programPromise;
  const n = features.tokens.length;
  const D = model.hidden;
  const W = 3 * D;
  const O = model.out;
  if (n === 0) return new Float32Array(0);

  const flat = new Uint32Array(n * model.width);
  for (const [i, row] of features.rows.entries()) flat.set(row, i * model.width);

  const params = program.buffer(
    "params",
    packParams(model, n),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const featureBuf = program.buffer("features", flat);
  const weights = program.buffer("weights", model.weights);
  const state = program.buffer("state", new Float32Array(n * D + n * W + model.head));
  const scan = program.buffer("scan", new Float32Array(6 * n * D));
  const logits = program.buffer(
    "logits",
    new Float32Array(n * O),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const out = await program.run(
    [params, featureBuf, weights, state, scan, logits],
    dispatch(model, n),
    logits,
  );
  return out.subarray(0, n * O);
}
