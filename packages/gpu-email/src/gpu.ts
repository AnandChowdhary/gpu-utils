import { createProgram, type FeatureRows, type Program } from "@gpu-utils/runtime";
import { logitWidth, type Model } from "./model.ts";
import shader from "./shader.wgsl";

let programPromise: Promise<Program> | undefined;

/** Compile-time widths baked into shader.wgsl; the manifest must agree. */
const SHADER_DIM = 48;
const SHADER_HEAD = 64;
const SHADER_KINDS = 8;
const SHADER_BIO = 15;
const SHADER_LAYERS = 6;

function offsetOf(model: Model, name: string): number {
  const entry = model.manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`unknown tensor ${name}`);
  return entry.offset;
}

/** WebGPU forward pass. Produces the same [n, KINDS + BIO] logits as forwardCpu. */
export async function forwardGpu(model: Model, features: FeatureRows): Promise<Float32Array> {
  const m = model.manifest;
  if (
    m.hidden !== SHADER_DIM ||
    m.head !== SHADER_HEAD ||
    m.labels.length !== SHADER_KINDS ||
    m.fields.length !== SHADER_BIO ||
    m.dilations.length !== SHADER_LAYERS
  ) {
    throw new Error("gpu-email: shader constants do not match the model manifest");
  }
  const entries = ["embed", ...Array.from({ length: SHADER_LAYERS }, (_, i) => `conv${i}`), "head"];
  programPromise ??= createProgram(shader, entries);
  const program = await programPromise;
  const n = features.tokens.length;
  const W = logitWidth(model);
  if (n === 0) return new Float32Array(0);
  const slots = m.slots;

  const flat = new Uint32Array(n * slots);
  for (const [i, row] of features.rows.entries()) flat.set(row, i * slots);

  const offsets = new Uint32Array(7 + 3 * SHADER_LAYERS);
  offsets.set([
    offsetOf(model, "emb"),
    offsetOf(model, "head.w"),
    offsetOf(model, "head.b"),
    offsetOf(model, "kind.w"),
    offsetOf(model, "kind.b"),
    offsetOf(model, "bio.w"),
    offsetOf(model, "bio.b"),
  ]);
  for (let l = 0; l < SHADER_LAYERS; l++) {
    offsets[7 + 3 * l] = offsetOf(model, `conv${l}.w`);
    offsets[8 + 3 * l] = offsetOf(model, `conv${l}.b`);
    offsets[9 + 3 * l] = m.dilations[l]!;
  }

  const params = program.buffer(
    "params",
    new Uint32Array([n, slots, SHADER_LAYERS, 0]),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const offsetsBuf = program.buffer("offsets", offsets);
  const featureBuf = program.buffer("features", flat);
  const weights = program.buffer("weights", model.weights);
  const state = program.buffer("state", new Float32Array((SHADER_LAYERS + 1) * n * SHADER_DIM));
  const logits = program.buffer(
    "logits",
    new Float32Array(n * W),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );

  const channelGroups = Math.ceil((n * SHADER_DIM) / 64);
  const tokenGroups = Math.ceil(n / 64);
  const out = await program.run(
    [params, offsetsBuf, featureBuf, weights, state, logits],
    [
      { entry: "embed", workgroups: [channelGroups] },
      ...Array.from({ length: SHADER_LAYERS }, (_, l) => ({
        entry: `conv${l}`,
        workgroups: [channelGroups] as [number],
      })),
      { entry: "head", workgroups: [tokenGroups] },
    ],
    logits,
  );
  return out.subarray(0, n * W);
}
