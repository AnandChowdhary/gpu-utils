import { type FeatureRows, type Token, viterbi } from "@gpu-utils/runtime";
import {
  applyVariants,
  correctWords,
  emit,
  isValidClass,
  literalClass,
  parseValue,
  resolveVariant,
  standalone,
  T,
  type Value,
  wordsOf,
} from "./compile.ts";
import type { Model } from "./model.ts";

export interface Diagnostic {
  message: string;
  start: number;
  end: number;
}

export interface Group {
  /** UTF-16 offsets of the phrase segment, half-open. */
  span: { start: number; end: number };
  text: string;
  classes: string[];
  variant?: string;
}

export interface TailwindResult {
  /** Deduplicated classes in phrase order, all validated against the Tailwind v4 vocabulary. */
  classes: string[];
  groups: Group[];
  diagnostics: Diagnostic[];
  /** Per-token role labels (debugging aid). */
  labels: string[];
  tokens: { text: string; start: number; end: number }[];
}

interface Span {
  role: "PROP" | "VAL" | "VAR" | "SEP" | "NEG";
  first: number;
  last: number;
  text: string;
  boundary: boolean;
}

function transitions(labels: string[]): Float32Array {
  const k = labels.length;
  const t = new Float32Array(k * k);
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      const to = labels[b]!;
      if (to.startsWith("I-")) {
        const from = labels[a]!;
        const ok = from === to || from === `B-${to.slice(2)}`;
        t[a * k + b] = ok ? 0 : Number.NEGATIVE_INFINITY;
      }
    }
  }
  return t;
}

function toSpans(tokens: Token[], labels: string[], boundary: boolean[]): Span[] {
  const spans: Span[] = [];
  let cur: Span | null = null;
  const close = () => {
    if (!cur) return;
    // trim trailing whitespace tokens
    while (cur.last > cur.first && /^\s+$/.test(tokens[cur.last]!.text)) cur.last--;
    cur.text = tokens
      .slice(cur.first, cur.last + 1)
      .map((t) => t.text)
      .join("");
    spans.push(cur);
    cur = null;
  };
  for (let i = 0; i < tokens.length; i++) {
    const lab = labels[i]!;
    const isSpace = /^\s+$/.test(tokens[i]!.text);
    if (lab === "O") {
      if (cur && isSpace) continue; // spaces inside a span may be tagged O; keep going
      close();
      continue;
    }
    if (lab === "SEP" || lab === "NEG") {
      if (isSpace) continue;
      close();
      spans.push({ role: lab, first: i, last: i, text: tokens[i]!.text, boundary: boundary[i]! });
      continue;
    }
    const role = lab.slice(2) as Span["role"];
    if (lab.startsWith("B-") || !cur || cur.role !== role) {
      if (isSpace && !cur) continue;
      close();
      cur = { role, first: i, last: i, text: "", boundary: boundary[i]! };
    } else {
      cur.last = i;
    }
  }
  close();
  return spans;
}

function segmentsOf(spans: Span[]): Span[][] {
  const segs: Span[][] = [];
  let cur: Span[] = [];
  for (const s of spans) {
    if (s.role === "SEP") {
      if (cur.length) segs.push(cur);
      cur = [];
      continue;
    }
    if (s.boundary && cur.length && s.role !== "NEG") {
      segs.push(cur);
      cur = [];
    }
    cur.push(s);
  }
  if (cur.length) segs.push(cur);
  return segs;
}

function lookupProp(text: string): string | null {
  const words = wordsOf(text);
  const direct = T.props[words.join(" ")];
  if (direct) return direct;
  const fixed = correctWords(words);
  if (fixed) return T.props[fixed.join(" ")] ?? null;
  return null;
}

function lookupValue(text: string): Value | null {
  const v = parseValue(text);
  if (v) return v;
  const lit = literalClass(text);
  if (lit) return { kind: "lit", value: lit, intensity: 0 };
  const fixed = correctWords(wordsOf(text));
  return fixed ? parseValue(fixed.join(" ")) : null;
}

function lookupVariant(text: string): string | null {
  const words = wordsOf(text);
  // correct typos first: a misspelt keyword would otherwise silently change the variant
  return resolveVariant(correctWords(words) ?? words);
}

interface Unit {
  span: Span;
  /** position among the segment's non-O spans */
  index: number;
  key?: string;
  value?: Value | null;
  neg: boolean;
}

function compileSegment(seg: Span[], tokens: Token[], diagnostics: Diagnostic[]): Group | null {
  const first = tokens[seg[0]!.first]!;
  const last = tokens[seg[seg.length - 1]!.last]!;
  const span = { start: first.start, end: last.end };
  const text = tokens
    .slice(seg[0]!.first, seg[seg.length - 1]!.last + 1)
    .map((t) => t.text)
    .join("");
  const diag = (message: string, s: Span) =>
    diagnostics.push({ message, start: tokens[s.first]!.start, end: tokens[s.last]!.end });

  const variants: string[] = [];
  const props: Unit[] = [];
  const vals: Unit[] = [];
  let pendingNeg = false;
  let index = 0;
  for (const s of seg) {
    if (s.role === "NEG") {
      pendingNeg = true;
      continue;
    }
    if (s.role === "VAR") {
      const v = lookupVariant(s.text);
      if (v) variants.push(v);
      else diag(`unknown variant "${s.text}"`, s);
      index++;
      continue;
    }
    if (s.role === "PROP") {
      const key = lookupProp(s.text);
      if (key) props.push({ span: s, index, key, neg: pendingNeg });
      else diag(`unknown property "${s.text}"`, s);
    } else {
      const value = lookupValue(s.text);
      if (value) vals.push({ span: s, index, value, neg: pendingNeg });
      else diag(`unknown value "${s.text}"`, s);
    }
    pendingNeg = false;
    index++;
  }

  // pair every value with the nearest compatible property, measured in spans; ties go to
  // the preceding property for numbers ("gap 4") and the following one for adjectives
  // ("blue background")
  const NUMERIC = new Set(["num", "unit", "pct", "frac"]);
  const assigned = new Map<Unit, Unit[]>();
  const loose: Unit[] = [];
  for (const v of vals) {
    let best: Unit | null = null;
    let bestDist = Number.POSITIVE_INFINITY;
    if (v.value!.kind !== "lit") {
      for (const p of props) {
        if (emit(p.key!, v.value!, false).length === 0) continue;
        const before = p.index < v.index;
        const dist = Math.abs(p.index - v.index);
        const prefer = NUMERIC.has(v.value!.kind) ? before : !before;
        if (dist < bestDist || (dist === bestDist && prefer)) {
          best = p;
          bestDist = dist;
        }
      }
    }
    if (best) {
      const list = assigned.get(best) ?? [];
      list.push(v);
      assigned.set(best, list);
    } else loose.push(v);
  }

  const classes: string[] = [];
  const push = (cs: string[], s: Span) => {
    for (const c of cs) {
      if (!isValidClass(c)) {
        diag(`dropped invalid class "${c}"`, s);
        continue;
      }
      if (!classes.includes(c)) classes.push(c);
    }
  };
  // emit in phrase order
  const units = [...props, ...loose].sort((a, b) => a.index - b.index);
  for (const u of units) {
    if (u.key !== undefined) {
      const values = assigned.get(u) ?? [];
      if (values.length === 0) {
        const cs = emit(u.key, null, u.neg);
        if (cs.length === 0 && !u.key.startsWith("preset:"))
          diag(`"${u.span.text}" needs a value`, u.span);
        push(cs, u.span);
      } else {
        for (const v of values) push(emit(u.key, v.value!, u.neg || v.neg), v.span);
      }
    } else {
      const cs = standalone(u.value!, u.neg);
      if (cs.length === 0) diag(`"${u.span.text}" needs a property`, u.span);
      push(cs, u.span);
    }
  }
  if (classes.length === 0 && variants.length > 0) {
    diagnostics.push({ message: `variant "${text}" has no classes to apply to`, ...span });
    return null;
  }
  if (classes.length === 0) {
    if (props.length || vals.length)
      diagnostics.push({ message: `no classes for "${text}"`, ...span });
    return null;
  }
  const final = applyVariants(classes, variants);
  const group: Group = { span, text, classes: final };
  if (variants.length) group.variant = final[0]!.slice(0, final[0]!.lastIndexOf(":"));
  return group;
}

/** Viterbi over role logits, then the deterministic compiler. */
export function decode(model: Model, features: FeatureRows, logits: Float32Array): TailwindResult {
  const labelsList = model.manifest.labels;
  const k = labelsList.length;
  const O = model.out;
  const n = features.tokens.length;
  const roles = new Float32Array(n * k);
  const boundary: boolean[] = [];
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < k; j++) roles[t * k + j] = logits[t * O + j]!;
    boundary.push(logits[t * O + O - 1]! > 0);
  }
  const path = viterbi(roles, n, k, transitions(labelsList));
  const labels = Array.from(path, (i) => labelsList[i] ?? "O");
  const spans = toSpans(features.tokens, labels, boundary);
  const diagnostics: Diagnostic[] = [];
  const groups: Group[] = [];
  for (const seg of segmentsOf(spans)) {
    const g = compileSegment(seg, features.tokens, diagnostics);
    if (g) groups.push(g);
  }
  const classes: string[] = [];
  for (const g of groups) for (const c of g.classes) if (!classes.includes(c)) classes.push(c);
  return {
    classes,
    groups,
    diagnostics,
    labels,
    tokens: features.tokens.map(({ text, start, end }) => ({ text, start, end })),
  };
}

/** Compile from gold labels (used by the oracle test to check compiler/generator parity). */
export function decodeLabels(
  features: FeatureRows,
  labels: string[],
  boundary: boolean[],
): TailwindResult {
  const spans = toSpans(features.tokens, labels, boundary);
  const diagnostics: Diagnostic[] = [];
  const groups: Group[] = [];
  for (const seg of segmentsOf(spans)) {
    const g = compileSegment(seg, features.tokens, diagnostics);
    if (g) groups.push(g);
  }
  const classes: string[] = [];
  for (const g of groups) for (const c of g.classes) if (!classes.includes(c)) classes.push(c);
  return {
    classes,
    groups,
    diagnostics,
    labels,
    tokens: features.tokens.map(({ text, start, end }) => ({ text, start, end })),
  };
}
