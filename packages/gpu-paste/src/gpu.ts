import { createProgram, type Program } from "@gpu-utils/runtime";
import { forwardCpu, type Logits } from "./cpu.ts";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

const ENTRIES = ["embed", "gates", "scan_local", "scan_fixup", "mix", "pool", "head"];
const DIM = 32;
const MIX = 48;
const KIND_HIDDEN = 32;
const CHUNK = 256;
const TENSORS = [
  "embed",
  "gate_f.w",
  "gate_f.b",
  "gate_b.w",
  "gate_b.b",
  "mix.w",
  "mix.b",
  "head.w",
  "head.b",
  "out.w",
  "out.b",
  "kind1.w",
  "kind1.b",
  "kind2.w",
  "kind2.b",
];

let programPromise: Promise<Program> | undefined;

/** Uniform block matching `struct Params` in shader.wgsl. */
export function packParams(model: Model, n: number): Uint32Array {
  const m = model.manifest;
  if (m.dim !== DIM || m.mix !== MIX || m.kindHidden !== KIND_HIDDEN) {
    throw new Error("shader.wgsl constants do not match model/manifest.json");
  }
  const offsets = TENSORS.map((name) => {
    const t = m.tensors.find((e) => e.name === name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return t.offset;
  });
  const out = new Uint32Array(24);
  out.set([n, Math.ceil(n / CHUNK), m.labels.length, m.kinds.length, m.featureCount, ...offsets]);
  return out;
}

/** WebGPU forward pass. Produces the same logits as forwardCpu (see training/tests/test_wgsl.py). */
export async function forwardGpu(model: Model, rows: number[][]): Promise<Logits> {
  const n = rows.length;
  if (n === 0) return forwardCpu(model, rows);
  programPromise ??= createProgram(shader, ENTRIES);
  const program = await programPromise;
  const F = model.manifest.featureCount;
  const L = model.manifest.labels.length;
  const K = model.manifest.kinds.length;

  const flat = new Uint32Array(n * F);
  for (let t = 0; t < n; t++) flat.set(rows[t]!, t * F);

  const params = program.buffer(
    "params",
    packParams(model, n),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const featureBuf = program.buffer("features", flat);
  const weights = program.buffer("weights", model.weights);
  const state = program.buffer("state", new Float32Array(n * (DIM + MIX) + MIX + KIND_HIDDEN));
  const scan = program.buffer("scan", new Float32Array(6 * n * DIM));
  const logits = program.buffer(
    "logits",
    new Float32Array(n * L + K),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );

  const out = await program.run(
    [params, featureBuf, weights, state, scan, logits],
    [
      { entry: "embed", workgroups: [Math.ceil((n * DIM) / 64)] },
      { entry: "gates", workgroups: [Math.ceil((2 * n * DIM) / 64)] },
      { entry: "scan_local", workgroups: [Math.ceil(n / CHUNK), 2 * DIM] },
      { entry: "scan_fixup", workgroups: [Math.ceil((2 * n * DIM) / 64)] },
      { entry: "mix", workgroups: [n] },
      { entry: "pool", workgroups: [1] },
      { entry: "head", workgroups: [n] },
    ],
    logits,
  );
  return { span: out.slice(0, n * L), kind: out.slice(n * L, n * L + K) };
}
