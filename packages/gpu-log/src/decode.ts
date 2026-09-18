import {
  type BioSpan,
  bioStartMask,
  bioToSpans,
  bioTransitions,
  type Token,
  viterbi,
} from "@gpu-utils/runtime";
import { parseJsonLine } from "./fastpath.ts";
import type { Model } from "./model.ts";
import { normalizeLevel, normalizeTimestamp } from "./normalize.ts";
import type { LogFrame, LogLine } from "./types.ts";

export interface Decoder {
  labels: string[];
  kinds: string[];
  transitions: Float32Array;
  start: Float32Array;
}

export function makeDecoder(model: Model): Decoder {
  const labels = model.manifest.labels;
  return {
    labels,
    kinds: model.manifest.kinds,
    transitions: bioTransitions(labels),
    start: bioStartMask(labels),
  };
}

function stripQuotes(s: string): string {
  const t = s.trim();
  if (
    t.length >= 2 &&
    ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))
  ) {
    return t.slice(1, -1);
  }
  return t;
}

const EXT_LANG: [RegExp, string][] = [
  [/\.(java|kt|kts|scala|groovy)$/i, "java"],
  [/\.(py|pyw|pyx)$|^<(stdin|string|module)>$/i, "python"],
  [/\.(m?[jt]sx?|cjs|mts|cts)$|^node:|^webpack|^file:\/\/|^https?:\/\//i, "javascript"],
  [/\.go$|\.s$/i, "go"],
  [/\.rs$/i, "rust"],
  [/\.(cs|vb|fs)$/i, "csharp"],
  [/\.(rb|rake|erb)$/i, "ruby"],
  [/\.(php|phtml|inc)$/i, "php"],
];

/** Language of a frame line from its shape (deterministic; falls back to the file extension). */
export function frameLanguage(text: string, frame: LogFrame): string | undefined {
  const t = text.trim();
  if (/^File ".*", line \d+/.test(t)) return "python";
  if (/^at .+\((?:.+\.(?:java|kt|kts|scala|groovy):\d+|Native Method|Unknown Source)\)/.test(t))
    return "java";
  if (
    /^at .+\) in .+:line \d+$/.test(t) ||
    /^at [\w.`<>|+$]+\((?:[A-Z][\w.`[\]]* \w+(?:, )?)*\)$/.test(t)
  ) {
    return "csharp";
  }
  if (/^(async )?at .*:\d+:\d+\)?$|^at .*\(<anonymous>\)|^\S+@\S+:\d+:\d+$/.test(t))
    return "javascript";
  if (/^#\d+ .*\(\d+\): |^#\d+ \[internal function\]/.test(t)) return "php";
  if (/^(from )?.+\.rb:\d+:in [`']/.test(t)) return "ruby";
  if (/\.go:\d+( \+0x[0-9a-f]+)?$|^(created by )?[\w./-]+\.[\w.()*]+\(.*\)$/.test(t)) return "go";
  if (/^\d+: |^at .+\.rs:\d+/.test(t) || /::/.test(frame.function ?? "")) return "rust";
  for (const [re, lang] of EXT_LANG) if (frame.file && re.test(frame.file)) return lang;
  return undefined;
}

/** HTTP access logs have positional values after the request line; name the first four. */
const ACCESS_KEYS = ["status", "bytes", "referrer", "user_agent"];

/**
 * Turns role spans (UTF-16 offsets into `text`) and the kind logits for one line into the
 * typed LogLine. All semantics (which span wins, normalisation, frame language, positional
 * kv keys, JSON payload merging) live here.
 */
export function compileLine(
  text: string,
  spans: BioSpan[],
  kindLogits: ArrayLike<number>,
  kinds: string[],
  index: number,
  span: [number, number],
): LogLine {
  const line: LogLine = { line: index, span, kind: "entry", kv: [] };
  let best = 0;
  for (let k = 1; k < kindLogits.length; k++) if (kindLogits[k]! > kindLogits[best]!) best = k;
  line.kind = (kinds[best] as LogLine["kind"]) ?? "entry";

  let msgStart = -1;
  let msgEnd = -1;
  let pendingKey: string | undefined;
  const values: string[] = [];
  const frame: LogFrame = {};
  for (const s of spans) {
    const raw = text.slice(s.start, s.end);
    switch (s.label) {
      case "TS":
        if (!line.timestamp) {
          const iso = normalizeTimestamp(raw);
          line.timestamp = iso ? { text: raw, iso } : { text: raw };
        }
        break;
      case "LEVEL":
        if (!line.level) {
          const level = normalizeLevel(raw);
          if (level) line.level = level;
        }
        break;
      case "SOURCE":
        if (!line.source) line.source = raw;
        break;
      case "HOST":
        if (!line.host) line.host = raw;
        break;
      case "THREAD":
        if (!line.thread) line.thread = raw;
        break;
      case "KEY":
        if (pendingKey !== undefined) line.kv.push({ key: pendingKey, value: "" });
        pendingKey = raw;
        break;
      case "VALUE":
        if (pendingKey !== undefined) {
          line.kv.push({ key: pendingKey, value: stripQuotes(raw) });
          pendingKey = undefined;
        } else values.push(stripQuotes(raw));
        break;
      case "MSG":
        if (msgStart < 0) msgStart = s.start;
        msgEnd = s.end;
        break;
      case "FN":
        if (!frame.function) frame.function = raw;
        break;
      case "FILE":
        if (!frame.file) frame.file = raw;
        break;
      case "LINE":
        if (frame.line === undefined && /^\d+$/.test(raw)) frame.line = Number(raw);
        break;
      case "COL":
        if (frame.column === undefined && /^\d+$/.test(raw)) frame.column = Number(raw);
        break;
    }
  }
  if (pendingKey !== undefined) line.kv.push({ key: pendingKey, value: "" });

  if (msgStart >= 0) {
    let message = text.slice(msgStart, msgEnd).trim();
    if (message.startsWith('"') && message.endsWith('"') && message.length >= 2)
      message = message.slice(1, -1);
    line.message = message;
    // A JSON payload as the message (docker/k8s wrapping a JSON logger): merge its fields.
    if (message.startsWith("{") && message.endsWith("}")) {
      const fields = parseJsonLine(message);
      if (fields) {
        if (!line.timestamp && fields.timestamp) line.timestamp = fields.timestamp;
        if (!line.level && fields.level) line.level = fields.level;
        if (!line.source && fields.source) line.source = fields.source;
        if (!line.thread && fields.thread) line.thread = fields.thread;
        if (fields.message !== undefined) line.message = fields.message;
        line.kv.push(...fields.kv);
      }
    }
  }
  if (values.length) {
    const isRequest = line.message !== undefined && /^[A-Z]{3,7} \S+ HTTP\/\d/.test(line.message);
    values.forEach((value, i) => {
      const key = isRequest && i < ACCESS_KEYS.length ? ACCESS_KEYS[i]! : `_${i + 1}`;
      line.kv.push({ key, value });
    });
  }
  // Level from a kv pair when the prefix had none (heroku `at=info`, `level=warn` tails).
  if (!line.level) {
    const kv = line.kv.find((p) => /^(level|lvl|severity|at)$/i.test(p.key));
    if (kv) {
      const level = normalizeLevel(kv.value);
      if (level) line.level = level;
    }
  }
  if (frame.function || frame.file || frame.line !== undefined) {
    const lang = frameLanguage(text, frame);
    if (lang) frame.language = lang;
    line.frame = frame;
  }
  return line;
}

/** Constrained Viterbi over one line's tag logits, then the compiler. */
export function decodeLine(
  dec: Decoder,
  tags: Float32Array,
  pooled: ArrayLike<number>,
  text: string,
  tokens: Token[],
  index: number,
  span: [number, number],
): LogLine {
  const k = dec.labels.length;
  const n = tokens.length;
  const em = Float32Array.from(tags);
  for (let j = 0; j < k && n > 0; j++) em[j]! += dec.start[j]!;
  const ids = viterbi(em, n, k, dec.transitions);
  const spans = bioToSpans(ids, dec.labels, tokens);
  return compileLine(text, spans, pooled, dec.kinds, index, span);
}

/** Two-line frames (Go, Rust): copy the function of a FN-only frame into the FILE-only frame below it. */
export function mergeTwoLineFrames(lines: LogLine[]): void {
  for (let i = 1; i < lines.length; i++) {
    const cur = lines[i]!;
    const prev = lines[i - 1]!;
    if (cur.kind !== "frame" || prev.kind !== "frame" || !cur.frame || !prev.frame) continue;
    if (
      cur.frame.function === undefined &&
      cur.frame.file !== undefined &&
      prev.frame.function !== undefined &&
      prev.frame.file === undefined
    ) {
      cur.frame.function = prev.frame.function;
      if (!cur.frame.language && prev.frame.language) cur.frame.language = prev.frame.language;
    }
  }
}
