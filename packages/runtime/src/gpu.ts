/**
 * Runs the canonical WGSL kernels for the two model families on top of createProgram.
 * Output matches models.ts per sequence (parity is checked by the Python harness on
 * lavapipe in tooling/python/tests/test_wgsl_parity.py).
 *
 *   const program = await createProgram(scanTaggerShader, scanTaggerEntries(layers));
 *   const { tags, pooled } = await runScanTagger(program, model, [rows1, rows2]);
 */
import { grid, packRows, taggerParams, tensorOffsets, unpackLogits } from "./batch.ts";
import { convTensorNames, scanTensorNames, type TaggerModel } from "./models.ts";
import type { Pass, Program } from "./program.ts";
import convTaggerShader from "./wgsl/conv_tagger.wgsl";
import scanTaggerShader from "./wgsl/scan_tagger.wgsl";

export { convTaggerShader, scanTaggerShader };

export const SCAN_TAGGER_MAX_LAYERS = 4;
export const CONV_TAGGER_MAX_BLOCKS = 8;

/** Entry points for a scan model: embed, (gates, scan, conv) per layer, pool, head, pooled. */
export function scanTaggerEntries(layers: number): string[] {
  if (layers > SCAN_TAGGER_MAX_LAYERS)
    throw new Error(`scan_tagger.wgsl supports at most ${SCAN_TAGGER_MAX_LAYERS} scan layers`);
  const entries = ["embed"];
  for (let l = 0; l < layers; l++) entries.push(`gates${l}`, `scan${l}`, `conv${l}`);
  entries.push("pool", "head", "pooled");
  return entries;
}

/** Entry points for a conv model: embed, block0..block{n-1}, head, pool. */
export function convTaggerEntries(blocks: number): string[] {
  if (blocks > CONV_TAGGER_MAX_BLOCKS)
    throw new Error(`conv_tagger.wgsl supports at most ${CONV_TAGGER_MAX_BLOCKS} blocks`);
  const entries = ["embed"];
  for (let i = 0; i < blocks; i++) entries.push(`block${i}`);
  entries.push("head", "pool");
  return entries;
}

export interface BatchOutput {
  /** Per sequence `[n, tags]` logits. */
  tags: Float32Array[];
  /** Per sequence `[pooled]` logits when the model has a pooled head. */
  pooled?: Float32Array[];
}

const UNIFORM = 0x40 | 0x8; // GPUBufferUsage.UNIFORM | COPY_DST (constants inlined: no WebGPU globals at import time)
const STORAGE = 0x80 | 0x8; // STORAGE | COPY_DST
const READBACK = 0x80 | 0x8 | 0x4; // STORAGE | COPY_DST | COPY_SRC

function limits(model: TaggerModel, scan: boolean): void {
  const m = model.manifest;
  const width = scan ? 2 * m.hidden : Math.max(m.hidden, m.embed ?? 0);
  if (width > 256 || m.head > 256) {
    throw new Error("canonical kernels support hidden*2/embed/head <= 256; write a package kernel");
  }
}

async function runFamily(
  program: Program,
  model: TaggerModel,
  batch: ArrayLike<ArrayLike<ArrayLike<number>>>,
  scan: boolean,
): Promise<BatchOutput> {
  const m = model.manifest;
  limits(model, scan);
  const n = batch.length;
  if (n === 0) return { tags: [] };
  const { rows, lengths, maxTokens } = packRows(batch, m.slots, m.paddingId);
  const layers = scan ? (m.scanLayers ?? 1) : (m.dilations ?? []).length;
  const names = scan ? scanTensorNames(m) : convTensorNames(m);
  // Head entries always occupy 8 table slots so conv_tagger.wgsl can find the dilations.
  const pad = m.pooled > 0 ? [] : [0, 0, 0, 0];
  const table = Uint32Array.from([...tensorOffsets(m, names), ...pad, ...(m.dilations ?? [])]);
  const params = taggerParams({
    batch: n,
    maxTokens,
    slots: m.slots,
    padding: m.paddingId,
    embed: m.embed ?? m.hidden,
    hidden: m.hidden,
    head: m.head,
    tags: m.tags,
    pooled: m.pooled,
    layers,
    taps: m.convTaps ?? 5,
  });
  const width = scan ? 2 * m.hidden : m.hidden;
  const scratchSize = scan ? 4 * n * maxTokens * width + n * width : 2 * n * maxTokens * width;
  const scratch = new Float32Array(Math.max(1, scratchSize));
  const logits = new Float32Array(Math.max(1, n * maxTokens * m.tags + n * m.pooled));
  const bindings = [
    program.buffer("params", params, UNIFORM),
    program.buffer("table", table, STORAGE),
    program.buffer("weights", model.weights, STORAGE),
    program.buffer("rows", rows, STORAGE),
    program.buffer("lengths", lengths, STORAGE),
    program.buffer("scratch", scratch, STORAGE),
    program.buffer("logits", logits, READBACK),
  ];
  const passes: Pass[] = scan
    ? scanTaggerPasses(layers, n * maxTokens, n)
    : convTaggerPasses(layers, n * maxTokens, n);
  const out = await program.run(bindings, passes, bindings[6]!);
  return unpackLogits(out, lengths, maxTokens, m.tags, m.pooled);
}

/** Dispatch list for scan_tagger.wgsl: per-token passes on a wrapped grid, scan/pool/pooled per sequence. */
export function scanTaggerPasses(layers: number, tokens: number, batch: number): Pass[] {
  const g = grid(tokens);
  const passes: Pass[] = [{ entry: "embed", workgroups: g }];
  for (let l = 0; l < layers; l++) {
    passes.push(
      { entry: `gates${l}`, workgroups: g },
      { entry: `scan${l}`, workgroups: [batch] },
      { entry: `conv${l}`, workgroups: g },
    );
  }
  passes.push(
    { entry: "pool", workgroups: [batch] },
    { entry: "head", workgroups: g },
    { entry: "pooled", workgroups: [batch] },
  );
  return passes;
}

/** Dispatch list for conv_tagger.wgsl: per-token passes on a wrapped grid, pool per sequence. */
export function convTaggerPasses(blocks: number, tokens: number, batch: number): Pass[] {
  const g = grid(tokens);
  const passes: Pass[] = [{ entry: "embed", workgroups: g }];
  for (let i = 0; i < blocks; i++) passes.push({ entry: `block${i}`, workgroups: g });
  passes.push({ entry: "head", workgroups: g }, { entry: "pool", workgroups: [batch] });
  return passes;
}

/** Scan family on the GPU. `program` = createProgram(scanTaggerShader, scanTaggerEntries(layers)). */
export function runScanTagger(
  program: Program,
  model: TaggerModel,
  batch: ArrayLike<ArrayLike<ArrayLike<number>>>,
): Promise<BatchOutput> {
  return runFamily(program, model, batch, true);
}

/** Conv family on the GPU. `program` = createProgram(convTaggerShader, convTaggerEntries(blocks)). */
export function runConvTagger(
  program: Program,
  model: TaggerModel,
  batch: ArrayLike<ArrayLike<ArrayLike<number>>>,
): Promise<BatchOutput> {
  return runFamily(program, model, batch, false);
}
