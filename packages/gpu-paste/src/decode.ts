import { bioStartMask, bioTransitions, type Token, viterbi } from "@gpu-utils/runtime";
import type { Logits } from "./cpu.ts";
import type { Model } from "./model.ts";
import { parseDate, parseMoney, parsePhone, type RuleSpan, ruleSpans } from "./rules.ts";

export interface ModelSpan extends RuleSpan {
  confidence: number;
}

let tableCache: { model: Model; transitions: Float32Array; start: Float32Array } | undefined;

/** BIO constraints from the runtime: I-x may only follow B-x or I-x, and no sequence starts inside a span. */
function tables(model: Model): { transitions: Float32Array; start: Float32Array } {
  if (tableCache?.model !== model) {
    const labels = model.manifest.labels;
    tableCache = {
      model,
      transitions: bioTransitions(labels),
      start: bioStartMask(labels),
    };
  }
  return tableCache;
}

function softmaxMax(row: Float32Array): number {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sum = 0;
  for (const v of row) sum += Math.exp(v - max);
  return 1 / sum;
}

/** Viterbi over BIO labels → character spans (UTF-16, whitespace-trimmed) with a mean-prob confidence. */
export function decodeSpans(model: Model, tokens: Token[], span: Float32Array): ModelSpan[] {
  const labels = model.manifest.labels;
  const k = labels.length;
  const n = tokens.length;
  if (n === 0) return [];
  const { transitions, start: startMask } = tables(model);
  const emissions = Float32Array.from(span);
  for (let l = 0; l < k; l++) emissions[l] = emissions[l]! + startMask[l]!;
  const path = viterbi(emissions, n, k, transitions);
  const out: ModelSpan[] = [];
  let start = -1;
  let kind = "";
  let probSum = 0;
  let count = 0;
  const flush = (endToken: number) => {
    if (start < 0) return;
    let s = start;
    let e = endToken - 1;
    while (s <= e && /^\s+$/.test(tokens[s]!.text)) s++;
    while (e >= s && /^\s+$/.test(tokens[e]!.text)) e--;
    if (s <= e) {
      out.push({
        kind,
        span: [tokens[s]!.start, tokens[e]!.end],
        confidence: count ? probSum / count : 0,
      });
    }
    start = -1;
  };
  for (let i = 0; i <= n; i++) {
    const name = i < n ? labels[path[i]!]! : "O";
    const isI = name.startsWith("I-") && name.slice(2) === kind && start >= 0;
    if (!isI) {
      flush(i);
      if (name.startsWith("B-")) {
        start = i;
        kind = name.slice(2);
        probSum = 0;
        count = 0;
      }
    }
    if (start >= 0 && i < n) {
      probSum += softmaxMax(span.subarray(i * k, (i + 1) * k));
      count++;
    }
  }
  return out;
}

export function softmax(logits: Float32Array): number[] {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Normalised value for a learned span, using the deterministic parsers where they apply. */
export function spanValue(kind: string, text: string): string | undefined {
  const t = text.trim();
  if (kind === "money") {
    const m = parseMoney(t);
    return m ? `${m.amount} ${m.currency}` : undefined;
  }
  if (kind === "date") return parseDate(t)?.iso;
  if (kind === "phone") {
    const p = parsePhone(t);
    return p ? (p.e164 ?? p.digits) : undefined;
  }
  return t.replace(/\s+/g, " ");
}

/**
 * Merges rule spans (authoritative) with model spans: any model span overlapping a rule
 * span is dropped, and the result is sorted by offset.
 */
export function mergeSpans(text: string, model: ModelSpan[]): RuleSpan[] {
  const rules = ruleSpans(text);
  const kept: RuleSpan[] = [...rules];
  for (const s of model) {
    if (rules.some((r) => s.span[0] < r.span[1] && s.span[1] > r.span[0])) continue;
    const value = spanValue(s.kind, text.slice(s.span[0], s.span[1]));
    const out: RuleSpan = { kind: s.kind, span: s.span };
    if (value !== undefined) out.value = value;
    kept.push(out);
  }
  return kept.sort((a, b) => a.span[0] - b.span[0] || a.span[1] - b.span[1]);
}

export { decodeSpans as decode, type Logits };
