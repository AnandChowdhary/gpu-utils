import { CharClass, viterbi } from "@gpu-utils/runtime";
import { EMAIL_RE, type EmailFeatures, type LineInfo, URL_RE } from "./features.ts";
import { logitWidth, type Model } from "./model.ts";

export type SegmentKind =
  | "reply"
  | "attribution"
  | "quote"
  | "signature"
  | "disclaimer"
  | "forward_header"
  | "greeting"
  | "closing";

export interface Segment {
  kind: SegmentKind;
  /** UTF-16 code unit offsets, half-open. */
  span: [number, number];
}

export interface Contact {
  name?: string;
  title?: string;
  company?: string;
  phone?: string[];
  email?: string[];
  url?: string[];
  address?: string;
  /** Span of the signature block the fields were read from. */
  span: [number, number];
}

export interface Diagnostics {
  backend: "cpu" | "webgpu";
  tokens: number;
  lines: number;
  /** Wall-clock milliseconds for featurize + forward + decode. */
  ms: number;
}

export interface EmailParseResult {
  segments: Segment[];
  /** The new content only (reply, greeting and closing lines), whitespace-trimmed. */
  reply: string;
  contact?: Contact;
  diagnostics: Diagnostics;
}

const KINDS: SegmentKind[] = [
  "reply",
  "attribution",
  "quote",
  "signature",
  "disclaimer",
  "forward_header",
  "greeting",
  "closing",
];
const REPLY = 0;
const ATTRIBUTION = 1;
const QUOTE = 2;
const SIGNATURE = 3;
const DISCLAIMER = 4;
const FORWARD = 5;
const GREETING = 6;
const CLOSING = 7;
const NEG = -1e9;

/**
 * Log-domain transition scores between consecutive non-blank lines (from → to).
 * Emissions are mean token log-probabilities, so a penalty of 1–3 nats needs real
 * model evidence to override. Rows/cols follow KINDS.
 */
const TRANSITIONS: number[][] = [
  //  reply  attrib quote  sig    discl  fwd    greet  close
  [0.0, 0.0, -0.5, -0.5, -0.5, 0.0, -2.5, 0.0], // reply →
  [-3.0, 0.0, 0.0, -4.0, -3.0, -2.0, -3.0, -4.0], // attribution →
  [-1.5, -0.5, 0.0, -2.0, -1.0, -1.0, -2.0, -2.0], // quote →
  [-3.0, -0.5, -1.5, 0.0, -0.5, -0.5, -4.0, -3.0], // signature →
  [-3.0, -0.5, -1.0, -1.0, 0.0, -0.5, -3.0, -3.0], // disclaimer →
  [-3.0, -3.0, 0.0, -3.0, -3.0, 0.0, -3.0, -3.0], // forward_header →
  [0.0, -1.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0], // greeting →
  [-1.5, -0.5, -1.0, 0.0, -0.5, -0.5, -3.0, -0.5], // closing →
];

function logSoftmax(logits: Float32Array, offset: number, k: number, out: Float64Array): void {
  let max = -Infinity;
  for (let j = 0; j < k; j++) if (logits[offset + j]! > max) max = logits[offset + j]!;
  let sum = 0;
  for (let j = 0; j < k; j++) sum += Math.exp(logits[offset + j]! - max);
  const lse = max + Math.log(sum);
  for (let j = 0; j < k; j++) out[j] = logits[offset + j]! - lse;
}

/** Decodes line kinds: token log-probs averaged per line, exact rules, then Viterbi. */
export function decodeLineKinds(
  model: Model,
  features: EmailFeatures,
  logits: Float32Array,
): Int32Array {
  const K = model.manifest.labels.length;
  const W = logitWidth(model);
  const lines = features.lines;
  const kinds = new Int32Array(lines.length).fill(-1);
  const active: number[] = [];
  for (let li = 0; li < lines.length; li++) if (!lines[li]!.blank) active.push(li);
  if (active.length === 0) return kinds;

  const emissions = new Float32Array(active.length * K);
  const lp = new Float64Array(K);
  let delimSeen = false;
  for (let a = 0; a < active.length; a++) {
    const line = lines[active[a]!]!;
    const acc = new Float64Array(K);
    let count = 0;
    for (let i = line.start; i < line.end; i++) {
      const t = features.tokens[i]!;
      if (t.cls === CharClass.Space || t.cls === CharClass.Newline) continue;
      logSoftmax(logits, i * W, K, lp);
      for (let j = 0; j < K; j++) acc[j] = acc[j]! + lp[j]!;
      count++;
    }
    if (count === 0) {
      for (let i = line.start; i < line.end; i++) {
        logSoftmax(logits, i * W, K, lp);
        for (let j = 0; j < K; j++) acc[j] = acc[j]! + lp[j]!;
        count++;
      }
    }
    const base = a * K;
    for (let j = 0; j < K; j++) emissions[base + j] = acc[j]! / Math.max(1, count);
    // Exact rules first.
    if (line.quotePrefixed) {
      for (let j = 0; j < K; j++) emissions[base + j] = j === QUOTE ? 0 : NEG;
    } else if (line.delimiter) {
      for (let j = 0; j < K; j++) emissions[base + j] = j === SIGNATURE ? 0 : NEG;
      delimSeen = true;
    } else if (delimSeen) {
      // After an RFC 3676 "-- " delimiter nothing new can be written by the author.
      emissions[base + REPLY] = NEG;
      emissions[base + GREETING] = NEG;
      emissions[base + CLOSING] = NEG;
    }
  }
  const trans = new Float32Array(K * K);
  for (let f = 0; f < K; f++) for (let t = 0; t < K; t++) trans[f * K + t] = TRANSITIONS[f]![t]!;
  const path = viterbi(emissions, active.length, K, trans);
  for (let a = 0; a < active.length; a++) kinds[active[a]!] = path[a]!;
  return kinds;
}

/** Merges consecutive lines of one kind (absorbing blank lines in between) into segments. */
export function buildSegments(lines: LineInfo[], kinds: Int32Array): Segment[] {
  const segments: Segment[] = [];
  let cur: { kind: number; start: number; end: number } | undefined;
  for (let li = 0; li < lines.length; li++) {
    const k = kinds[li]!;
    if (k < 0) continue;
    const line = lines[li]!;
    if (cur && cur.kind === k) {
      cur.end = line.charEnd;
    } else {
      if (cur) segments.push({ kind: KINDS[cur.kind]!, span: [cur.start, cur.end] });
      cur = { kind: k, start: line.charStart, end: line.charEnd };
    }
  }
  if (cur) segments.push({ kind: KINDS[cur.kind]!, span: [cur.start, cur.end] });
  return segments;
}

/** New content: reply, greeting and closing lines, blank runs collapsed, trimmed. */
export function buildReply(lines: LineInfo[], kinds: Int32Array): string {
  const out: string[] = [];
  for (let li = 0; li < lines.length; li++) {
    const k = kinds[li]!;
    if (k === REPLY || k === GREETING || k === CLOSING) {
      out.push(lines[li]!.text.trimEnd());
    } else if (out.length > 0 && out[out.length - 1] !== "") {
      out.push("");
    }
  }
  return out.join("\n").trim();
}

interface FieldSpan {
  field: string;
  start: number;
  end: number;
}

function bioTransitions(fields: string[]): Float32Array {
  const B = fields.length;
  const t = new Float32Array(B * B);
  for (let f = 0; f < B; f++) {
    for (let to = 0; to < B; to++) {
      const toLabel = fields[to]!;
      if (toLabel.startsWith("I-")) {
        const fromLabel = fields[f]!;
        const ok = fromLabel !== "O" && fromLabel.slice(2) === toLabel.slice(2);
        t[f * B + to] = ok ? 0 : NEG;
      } else {
        t[f * B + to] = 0;
      }
    }
  }
  return t;
}

/** BIO-decodes the contact head over one signature segment, then applies exact regexes. */
export function extractContact(
  model: Model,
  features: EmailFeatures,
  logits: Float32Array,
  text: string,
  span: [number, number],
): Contact | undefined {
  const K = model.manifest.labels.length;
  const fields = model.manifest.fields;
  const B = fields.length;
  const W = logitWidth(model);
  const tokens = features.tokens;
  let first = -1;
  let last = -1;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]!;
    if (t.end <= span[0] || t.start >= span[1]) continue;
    if (first < 0) first = i;
    last = i;
  }
  if (first < 0) return undefined;
  const n = last - first + 1;
  const em = new Float32Array(n * B);
  for (let i = 0; i < n; i++) {
    const src = (first + i) * W + K;
    for (let j = 0; j < B; j++) em[i * B + j] = logits[src + j]!;
  }
  const path = viterbi(em, n, B, bioTransitions(fields));
  const spans: FieldSpan[] = [];
  for (let i = 0; i < n; i++) {
    const label = fields[path[i]!]!;
    const t = tokens[first + i]!;
    if (label.startsWith("B-")) {
      spans.push({ field: label.slice(2), start: t.start, end: t.end });
    } else if (label.startsWith("I-") && spans.length > 0) {
      const s = spans[spans.length - 1]!;
      if (s.field === label.slice(2) && s.end >= t.start) s.end = t.end;
    }
  }
  // Exact regexes override the model for EMAIL and URL.
  const segText = text.slice(span[0], span[1]);
  const exact: FieldSpan[] = [];
  for (const m of segText.matchAll(new RegExp(EMAIL_RE.source, "g"))) {
    exact.push({ field: "EMAIL", start: span[0] + m.index, end: span[0] + m.index + m[0].length });
  }
  for (const m of segText.matchAll(new RegExp(URL_RE.source, "g"))) {
    let s = m[0];
    while (s.length > 0 && ".,;:!?".includes(s[s.length - 1]!)) s = s.slice(0, -1);
    const start = span[0] + m.index;
    if (exact.some((e) => start < e.end && start + s.length > e.start)) continue;
    exact.push({ field: "URL", start, end: start + s.length });
  }
  const kept = spans.filter(
    (s) =>
      s.field !== "EMAIL" &&
      s.field !== "URL" &&
      !exact.some((e) => s.start < e.end && s.end > e.start),
  );
  const all = [...kept, ...exact].sort((a, b) => a.start - b.start);
  const contact: Contact = { span };
  let any = false;
  const lists: Record<string, string[]> = { phone: [], email: [], url: [] };
  for (const s of all) {
    const value = text.slice(s.start, s.end).trim();
    if (!value) continue;
    if (s.field === "PHONE") {
      if ((value.match(/\d/g) ?? []).length < 5) continue;
      lists.phone!.push(value);
    } else if (s.field === "EMAIL") {
      lists.email!.push(value);
    } else if (s.field === "URL") {
      lists.url!.push(value);
    } else if (s.field === "ADDRESS") {
      contact.address = contact.address ? `${contact.address}, ${value}` : value;
    } else if (s.field === "NAME") {
      contact.name ??= value;
    } else if (s.field === "TITLE") {
      contact.title ??= value;
    } else if (s.field === "COMPANY") {
      contact.company ??= value;
    } else {
      continue;
    }
    any = true;
  }
  for (const key of ["phone", "email", "url"] as const) {
    const v = lists[key]!;
    if (v.length > 0) contact[key] = Array.from(new Set(v));
  }
  return any ? contact : undefined;
}

/**
 * Picks the author's own signature: the first signature segment after the first
 * new-content line (or the first signature at all when there is no new content).
 */
export function authorSignature(
  lines: LineInfo[],
  kinds: Int32Array,
  segments: Segment[],
): Segment | undefined {
  let firstReply = -1;
  for (let li = 0; li < lines.length; li++) {
    const k = kinds[li]!;
    if (k === REPLY || k === GREETING || k === CLOSING) {
      firstReply = lines[li]!.charStart;
      break;
    }
  }
  return segments.find((s) => s.kind === "signature" && s.span[0] >= firstReply);
}

export function decode(
  model: Model,
  features: EmailFeatures,
  logits: Float32Array,
  text: string,
): Omit<EmailParseResult, "diagnostics"> {
  const kinds = decodeLineKinds(model, features, logits);
  const segments = buildSegments(features.lines, kinds);
  const reply = buildReply(features.lines, kinds);
  const sig = authorSignature(features.lines, kinds, segments);
  const contact = sig ? extractContact(model, features, logits, text, sig.span) : undefined;
  return contact ? { segments, reply, contact } : { segments, reply };
}

export {
  ATTRIBUTION,
  CLOSING,
  DISCLAIMER,
  FORWARD,
  GREETING,
  KINDS as LINE_KINDS,
  QUOTE,
  REPLY,
  SIGNATURE,
};
