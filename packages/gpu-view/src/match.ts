/**
 * Tolerant surface-form matching of query tokens against a schema.
 * Mirrors training/gpu_view/match.py exactly (fixtures in test/fixtures/match.json).
 *
 * Matches n-grams (1..3 words) of letter/digit tokens, allowing "-" and "_" between
 * words. Quality tiers, best first: exact, stem (plural/inflection), prefix (>= 4
 * chars, unique), typo (>= 5 chars, one edit, unique). Multi-word spans use exact and
 * stem only. Both the featurizer and the compiler call this, so they always agree.
 */
import { CharClass, type Token, tokenize } from "@gpu-utils/runtime";

export type FieldKind = "text" | "number" | "date" | "enum" | "boolean";

export interface SchemaField {
  name: string;
  kind: FieldKind;
  aliases?: string[];
  values?: string[];
  /** Date field that bare time phrases ("this quarter") refer to when the schema has several. */
  primary?: boolean;
}

export interface Schema {
  fields: SchemaField[];
}

export const EXACT = 0;
export const STEM = 1;
export const INFLECT = 2;
export const PREFIX = 3;
export const TYPO = 4;
const MIN_PREFIX = 4;
const MIN_TYPO = 5;
const MAX_WORDS = 3;
const CONNECTORS = new Set(["-", "_"]);

export interface Entry {
  words: string[];
  field: number;
  kind: FieldKind;
  alias: boolean;
  value: number;
}

export interface Span {
  start: number;
  end: number;
  field: number;
  kind: FieldKind;
  quality: number;
  alias: boolean;
  value: number;
  owners: number[];
  /** Matched a boolean field through a negation prefix ("unopened" -> "opened"). */
  neg: boolean;
}

/** Negation prefixes stripped when matching a boolean field: "unarchived", "non-vip", "inactive". */
const NEG_PREFIXES = ["un", "non", "dis", "im", "ir", "in"];
const MIN_NEG_BASE = 3;

const isWord = (t: Token) => t.cls === CharClass.Letter || t.cls === CharClass.Digit;

/** Lower-cased letter/digit runs of a surface form; camelCase and snake_case split. */
export function wordsOf(surface: string): string[] {
  const text = surface.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  return tokenize(text)
    .filter(isWord)
    .map((t) => t.text.toLowerCase());
}

export function stem(word: string): string {
  const n = word.length;
  if (n >= 5 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (n >= 5 && word.endsWith("sses")) return word.slice(0, -2);
  if (n >= 5 && word.endsWith("es") && "sxz".includes(word[n - 3]!)) return word.slice(0, -2);
  if (n >= 6 && (word.endsWith("shes") || word.endsWith("ches"))) return word.slice(0, -2);
  if (n >= 4 && word.endsWith("s") && !word.endsWith("ss")) return word.slice(0, -1);
  return word;
}

/**
 * Stemmed base forms reachable by stripping comparative/superlative/participle endings:
 * "rated" and "rating" share "rat"/"rate"; "cheapest" reaches "cheap". Mirrors match.py.
 */
export function forms(word: string): Set<string> {
  const out = new Set([stem(word)]);
  const n = word.length;
  const cands: string[] = [];
  if (n >= 6 && word.endsWith("iest")) cands.push(`${word.slice(0, -4)}y`);
  if (n >= 6 && word.endsWith("est")) cands.push(word.slice(0, -3));
  if (n >= 5 && word.endsWith("ier")) cands.push(`${word.slice(0, -3)}y`);
  if (n >= 5 && word.endsWith("er")) cands.push(word.slice(0, -2));
  if (n >= 5 && word.endsWith("ied")) cands.push(`${word.slice(0, -3)}y`);
  if (n >= 5 && word.endsWith("ed")) cands.push(word.slice(0, -2), word.slice(0, -1));
  if (n >= 6 && word.endsWith("ing")) cands.push(word.slice(0, -3), `${word.slice(0, -3)}e`);
  for (const c of [...cands]) {
    if (c.length >= 3 && c[c.length - 1] === c[c.length - 2] && !"aeiou".includes(c[c.length - 1]!))
      cands.push(c.slice(0, -1));
  }
  for (const c of cands) if (c.length >= 3) out.add(stem(c));
  return out;
}

export function withinOneEdit(a: string, b: string): boolean {
  if (a === b || Math.abs(a.length - b.length) > 1) return false;
  if (a.length === b.length) {
    let diff = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
    return diff === 1;
  }
  const [short, long] = a.length < b.length ? [a, b] : [b, a];
  let i = 0;
  while (i < short.length && short[i] === long[i]) i++;
  return short.slice(i) === long.slice(i + 1);
}

export function fieldEntries(schema: Schema): Entry[] {
  const out: Entry[] = [];
  schema.fields.forEach((f, i) => {
    const forms: [string, boolean][] = [
      [f.name, false],
      ...(f.aliases ?? []).map((a): [string, boolean] => [a, true]),
    ];
    for (const [surface, alias] of forms) {
      const words = wordsOf(surface);
      if (words.length > 0 && words.length <= MAX_WORDS)
        out.push({ words, field: i, kind: f.kind, alias, value: -1 });
    }
  });
  return out;
}

export function enumEntries(schema: Schema): Entry[] {
  const out: Entry[] = [];
  schema.fields.forEach((f, i) => {
    (f.values ?? []).forEach((v, j) => {
      const words = wordsOf(v);
      if (words.length > 0 && words.length <= MAX_WORDS)
        out.push({ words, field: i, kind: f.kind, alias: false, value: j });
    });
  });
  return out;
}

/** Every token except whitespace/newline runs: the sequence the model sees. */
export function modelTokens(text: string): Token[] {
  return tokenize(text).filter((t) => t.cls !== CharClass.Space && t.cls !== CharClass.Newline);
}

export function wordPositions(tokens: Token[]): number[] {
  const out: number[] = [];
  tokens.forEach((t, i) => {
    if (isWord(t)) out.push(i);
  });
  return out;
}

function connected(tokens: Token[], a: number, b: number): boolean {
  for (let k = a + 1; k < b; k++) if (!CONNECTORS.has(tokens[k]!.text)) return false;
  return true;
}

const same = (a: string[], b: string[]) => a.length === b.length && a.every((w, i) => w === b[i]);

function matchAt(tokens: Token[], positions: number[], wi: number, entries: Entry[]): Span | null {
  const lowered = tokens.map((t) => t.text.toLowerCase());
  for (let n = MAX_WORDS; n >= 1; n--) {
    if (wi + n > positions.length) continue;
    const idx = positions.slice(wi, wi + n);
    let ok = true;
    for (let k = 0; k < n - 1; k++) if (!connected(tokens, idx[k]!, idx[k + 1]!)) ok = false;
    if (!ok) continue;
    const spanWords = idx.map((i) => lowered[i]!);
    const stems = spanWords.map(stem);
    const start = idx[0]!;
    const end = idx[n - 1]! + 1;
    for (const quality of [EXACT, STEM]) {
      const hits = entries.filter(
        (e) =>
          e.words.length === n &&
          (quality === EXACT ? same(e.words, spanWords) : same(e.words.map(stem), stems)),
      );
      if (hits.length > 0) {
        const first = hits[0]!;
        const owners = [...new Set(hits.map((e) => e.field))].sort((a, b) => a - b);
        return {
          start,
          end,
          field: first.field,
          kind: first.kind,
          quality,
          alias: first.alias,
          value: first.value,
          owners,
          neg: false,
        };
      }
    }
    if (n === 1) {
      const w = spanWords[0]!;
      if (w.length >= 5) {
        const qforms = forms(w);
        const hits = entries.filter(
          (e) => e.words.length === 1 && [...forms(e.words[0]!)].some((f) => qforms.has(f)),
        );
        if (hits.length > 0) {
          const e = hits[0]!;
          const owners = [...new Set(hits.map((h) => h.field))].sort((a, b) => a - b);
          return {
            start,
            end,
            field: e.field,
            kind: e.kind,
            quality: INFLECT,
            alias: e.alias,
            value: e.value,
            owners,
            neg: false,
          };
        }
      }
      const negated = matchNegated(w, entries);
      if (negated) return { ...negated, start, end };
      if (w.length >= MIN_PREFIX) {
        const hits = entries.filter(
          (e) =>
            e.words.length === 1 && e.words[0]!.length >= MIN_PREFIX && e.words[0]!.startsWith(w),
        );
        const keys = new Set(hits.map((e) => `${e.field}:${e.value}`));
        if (keys.size === 1) {
          const e = hits[0]!;
          return {
            start,
            end,
            field: e.field,
            kind: e.kind,
            quality: PREFIX,
            alias: e.alias,
            value: e.value,
            owners: [e.field],
            neg: false,
          };
        }
      }
      if (w.length >= MIN_TYPO) {
        const hits = entries.filter((e) => e.words.length === 1 && withinOneEdit(w, e.words[0]!));
        const keys = new Set(hits.map((e) => `${e.field}:${e.value}`));
        if (keys.size === 1) {
          const e = hits[0]!;
          return {
            start,
            end,
            field: e.field,
            kind: e.kind,
            quality: TYPO,
            alias: e.alias,
            value: e.value,
            owners: [e.field],
            neg: false,
          };
        }
      }
    }
  }
  return null;
}

/**
 * "unarchived" / "inactive" / "nonbillable": a boolean field word carrying a negation
 * prefix. Only boolean fields are tried, so ordinary words that happen to start with
 * "in" or "un" cannot be turned into a field by accident.
 */
function matchNegated(
  w: string,
  entries: Entry[],
): Omit<Span, "start" | "end"> | null {
  for (const prefix of NEG_PREFIXES) {
    if (!w.startsWith(prefix) || w.length - prefix.length < MIN_NEG_BASE) continue;
    const base = w.slice(prefix.length);
    const bforms = forms(base);
    const hits = entries.filter(
      (e) =>
        e.kind === "boolean" &&
        e.words.length === 1 &&
        (e.words[0] === base || [...forms(e.words[0]!)].some((f) => bforms.has(f))),
    );
    if (hits.length === 0) continue;
    const e = hits[0]!;
    return {
      field: e.field,
      kind: e.kind,
      quality: INFLECT,
      alias: e.alias,
      value: e.value,
      owners: [...new Set(hits.map((h) => h.field))].sort((a, b) => a - b),
      neg: true,
    };
  }
  return null;
}

/** Greedy left-to-right, longest-first matching. Spans never overlap. */
export function matchSpans(tokens: Token[], entries: Entry[]): Span[] {
  const positions = wordPositions(tokens);
  const spans: Span[] = [];
  let wi = 0;
  while (wi < positions.length) {
    const found = matchAt(tokens, positions, wi, entries);
    if (!found) {
      wi++;
      continue;
    }
    spans.push(found);
    while (wi < positions.length && positions[wi]! < found.end) wi++;
  }
  return spans;
}

/** Resolve an already-extracted phrase (compiler side). Indices are relative to the phrase. */
export function resolveWords(words: string[], entries: Entry[]): Span | null {
  const toks = modelTokens(words.join(" "));
  const positions = wordPositions(toks);
  if (positions.length === 0) return null;
  return matchAt(toks, positions, 0, entries);
}
