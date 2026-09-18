/**
 * gpu-log: universal log line parser. Timestamps, levels, sources, threads, key=value pairs,
 * messages and stack frames from any text log format, on WebGPU or the CPU.
 *
 * Pipeline per line: blank check → JSON / logfmt fast paths → (everything else) character-class
 * tokens → sparse hashed features → dilated-CNN tagger (batched across lines per dispatch) →
 * Viterbi over BIO tags → deterministic compiler (decode.ts).
 */
import { type Backend, hasWebGPU, type Token, tokenize } from "@gpu-utils/runtime";
import { type Batch, forwardCpu } from "./cpu.ts";
import { bioTransitions, decodeLine, mergeTwoLineFrames } from "./decode.ts";
import { fastPath } from "./fastpath.ts";
import { FEATURE_COUNT, writeTokenFeatures } from "./features.ts";
import { forwardGpu } from "./gpu.ts";
import { MODEL, type Model } from "./model.ts";
import type { LogLine, LogParseResult } from "./types.ts";

export { normalizeLevel, normalizeTimestamp } from "./normalize.ts";
export type {
  Level,
  LineKind,
  LogFrame,
  LogKv,
  LogLine,
  LogParseResult,
  LogTimestamp,
} from "./types.ts";

export interface LogParseOptions {
  /** "auto" uses WebGPU for large inputs when available and the CPU reference otherwise. */
  backend?: Backend;
  /** Skip the deterministic JSON / logfmt paths and send every line to the model. */
  fastPaths?: boolean;
  /** Maximum tokens per forward pass (one GPU dispatch); lines are never split. */
  batchTokens?: number;
  /**
   * Reuse model output for lines whose feature sequence was already seen in this call
   * (digits are collapsed by the hashing, so repeated templates hit). Exact, on by default.
   */
  memoize?: boolean;
}

/** Inputs with fewer model tokens than this run on the CPU under "auto": readback latency dominates. */
const GPU_MIN_TOKENS = 256;
const DEFAULT_BATCH_TOKENS = 1 << 17;

interface PendingLine {
  index: number;
  text: string;
  tokens: Token[];
  /** Per-token feature ids, also the memoisation key material. */
  features: Uint32Array;
  key?: string;
}

/** Two independent FNV-1a hashes over the feature ids: a 64-bit memo key. */
function featureKey(features: Uint32Array): string {
  let a = 0x811c9dc5;
  let b = 0x01000193;
  for (let i = 0; i < features.length; i++) {
    const v = features[i]!;
    a = Math.imul(a ^ v, 0x01000193) >>> 0;
    b = Math.imul(b ^ (v + 0x9e3779b9), 0x85ebca6b) >>> 0;
  }
  return `${features.length}:${a}:${b}`;
}

const TRANSITIONS = bioTransitions(MODEL.manifest.labels);

/** Splits text into lines, keeping UTF-16 spans (line breaks excluded). */
export function splitLines(text: string): [number, number][] {
  const spans: [number, number][] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c === 10) {
      const end = i > start && text.charCodeAt(i - 1) === 13 ? i - 1 : i;
      spans.push([start, end]);
      start = i + 1;
    }
  }
  if (start < text.length) spans.push([start, text.length]);
  return spans;
}

/** Packs pending lines into flat batches of at most `batchTokens` tokens (whole lines only). */
function packBatches(
  pending: PendingLine[],
  batchTokens: number,
): { batch: Batch; lines: PendingLine[] }[] {
  const out: { batch: Batch; lines: PendingLine[] }[] = [];
  let group: PendingLine[] = [];
  let count = 0;
  const flush = () => {
    if (group.length === 0) return;
    const features = new Uint32Array(count * FEATURE_COUNT);
    const lineId = new Uint32Array(count);
    const offsets: number[] = [0];
    let p = 0;
    for (const [li, line] of group.entries()) {
      features.set(line.features, p * FEATURE_COUNT);
      for (let i = 0; i < line.tokens.length; i++, p++) lineId[p] = li;
      offsets.push(p);
    }
    out.push({ batch: { n: count, features, lineId, offsets }, lines: group });
    group = [];
    count = 0;
  };
  for (const line of pending) {
    if (count > 0 && count + line.tokens.length > batchTokens) flush();
    group.push(line);
    count += line.tokens.length;
  }
  flush();
  return out;
}

export async function parse(text: string, options: LogParseOptions = {}): Promise<LogParseResult> {
  const model: Model = MODEL;
  const spans = splitLines(text);
  const lines: LogLine[] = new Array(spans.length);
  const stats: LogParseResult["stats"] = {
    json: 0,
    logfmt: 0,
    model: 0,
    memoized: 0,
    blank: 0,
    backend: "cpu",
  };
  const pending: PendingLine[] = [];
  const useFast = options.fastPaths ?? true;

  for (let i = 0; i < spans.length; i++) {
    const span = spans[i]!;
    const lineText = text.slice(span[0], span[1]);
    if (lineText.trim() === "") {
      lines[i] = { line: i, span, kind: "blank", kv: [] };
      stats.blank++;
      continue;
    }
    if (useFast) {
      const fast = fastPath(lineText);
      if (fast) {
        lines[i] = { line: i, span, kind: "entry", ...fast.fields };
        stats[fast.path]++;
        continue;
      }
    }
    const tokens = tokenize(lineText);
    const features = new Uint32Array(tokens.length * FEATURE_COUNT);
    for (let t = 0; t < tokens.length; t++)
      writeTokenFeatures(tokens, t, features, t * FEATURE_COUNT);
    pending.push({ index: i, text: lineText, tokens, features });
  }
  stats.model = pending.length;

  // Memoisation: lines with an identical feature sequence get identical logits.
  const waiting = new Map<string, PendingLine[]>();
  let unique: PendingLine[] = pending;
  if (options.memoize ?? true) {
    unique = [];
    for (const line of pending) {
      line.key = featureKey(line.features);
      const others = waiting.get(line.key);
      if (others) others.push(line);
      else {
        waiting.set(line.key, []);
        unique.push(line);
      }
    }
    stats.memoized = pending.length - unique.length;
  }

  if (unique.length > 0) {
    let total = 0;
    for (const p of unique) total += p.tokens.length;
    const backend = options.backend ?? "auto";
    const useGpu =
      backend === "webgpu" || (backend === "auto" && hasWebGPU() && total >= GPU_MIN_TOKENS);
    stats.backend = useGpu ? "webgpu" : "cpu";
    const batches = packBatches(unique, options.batchTokens ?? DEFAULT_BATCH_TOKENS);
    const W = model.tags + model.kinds;
    const emit = (line: PendingLine, logits: Float32Array) => {
      lines[line.index] = decodeLine(
        model,
        TRANSITIONS,
        logits,
        line.text,
        line.tokens,
        line.index,
        spans[line.index]!,
      );
    };
    for (const { batch, lines: group } of batches) {
      const logits = useGpu ? await forwardGpu(model, batch) : forwardCpu(model, batch);
      for (const [li, line] of group.entries()) {
        const slice = logits.subarray(batch.offsets[li]! * W, batch.offsets[li + 1]! * W);
        emit(line, slice);
        if (line.key !== undefined)
          for (const other of waiting.get(line.key) ?? []) emit(other, slice);
      }
    }
  }
  mergeTwoLineFrames(lines);
  return { lines, stats };
}
