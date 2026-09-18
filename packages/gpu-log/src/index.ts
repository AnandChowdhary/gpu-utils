/**
 * gpu-log: universal log line parser. Timestamps, levels, sources, hosts, threads, key=value
 * pairs, messages and stack frames from any text log format, on WebGPU or the CPU.
 *
 * Pipeline per line: blank check → JSON / logfmt fast paths → (everything else) character-class
 * tokens → sparse hashed features → shared conv-family tagger (many lines per dispatch) →
 * Viterbi over BIO tags → deterministic compiler (decode.ts).
 */
import { type Backend, hasWebGPU, type Token, tokenize } from "@gpu-utils/runtime";
import { forwardCpu } from "./cpu.ts";
import { decodeLine, makeDecoder, mergeTwoLineFrames } from "./decode.ts";
import { fastPath } from "./fastpath.ts";
import { featurizeTokens } from "./features.ts";
import { forwardGpuBatch } from "./gpu.ts";
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
  /** Upper bound on padded tokens (lines × longest line) per GPU dispatch; lines are never split. */
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
  rows: number[][];
  key?: string;
}

/** Two independent FNV-1a hashes over the feature ids: a 64-bit memo key. */
function featureKey(rows: number[][]): string {
  let a = 0x811c9dc5;
  let b = 0x01000193;
  for (const row of rows) {
    for (const v of row) {
      a = Math.imul(a ^ v, 0x01000193) >>> 0;
      b = Math.imul(b ^ (v + 0x9e3779b9), 0x85ebca6b) >>> 0;
    }
  }
  return `${rows.length}:${a}:${b}`;
}

const DECODER = makeDecoder(MODEL);

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

/**
 * Groups lines of similar length into GPU batches. The kernel pads every line in a batch to
 * the longest one, so lines are sorted by length and a batch closes when `lines × longest`
 * would exceed `batchTokens`.
 */
export function packBatches(pending: PendingLine[], batchTokens: number): PendingLine[][] {
  const sorted = [...pending].sort((a, b) => a.tokens.length - b.tokens.length);
  const out: PendingLine[][] = [];
  let group: PendingLine[] = [];
  let longest = 0;
  for (const line of sorted) {
    const n = Math.max(1, line.tokens.length);
    if (group.length > 0 && (group.length + 1) * Math.max(longest, n) > batchTokens) {
      out.push(group);
      group = [];
      longest = 0;
    }
    group.push(line);
    longest = Math.max(longest, n);
  }
  if (group.length) out.push(group);
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
    pending.push({ index: i, text: lineText, tokens, rows: featurizeTokens(tokens) });
  }
  stats.model = pending.length;

  // Memoisation: lines with an identical feature sequence get identical logits.
  const waiting = new Map<string, PendingLine[]>();
  let unique: PendingLine[] = pending;
  if (options.memoize ?? true) {
    unique = [];
    for (const line of pending) {
      line.key = featureKey(line.rows);
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
    const decodeOne = (line: PendingLine, tags: Float32Array, pooled: ArrayLike<number>) => {
      lines[line.index] = decodeLine(
        DECODER,
        tags,
        pooled,
        line.text,
        line.tokens,
        line.index,
        spans[line.index]!,
      );
    };
    const emit = (line: PendingLine, tags: Float32Array, pooled: ArrayLike<number>) => {
      decodeOne(line, tags, pooled);
      if (line.key !== undefined)
        for (const other of waiting.get(line.key) ?? []) decodeOne(other, tags, pooled);
    };
    if (useGpu) {
      for (const group of packBatches(unique, options.batchTokens ?? DEFAULT_BATCH_TOKENS)) {
        const out = await forwardGpuBatch(
          model,
          group.map((l) => l.rows),
        );
        for (const [i, line] of group.entries()) emit(line, out.tags[i]!, out.pooled?.[i] ?? []);
      }
    } else {
      for (const line of unique) {
        const out = forwardCpu(model, line.rows);
        emit(line, out.tags, out.pooled ?? []);
      }
    }
  }
  mergeTwoLineFrames(lines);
  return { lines, stats };
}
