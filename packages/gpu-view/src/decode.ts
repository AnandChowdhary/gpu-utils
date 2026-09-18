/**
 * Argmax decode of the tagger's logits, then the deterministic compiler that turns
 * token roles into a typed view spec. All semantics live here, not in the model:
 * field resolution (aliases, plurals, one-character typos), enum canonicalisation,
 * numbers with k/m suffixes, relative dates, operator phrases and negation. Anything
 * that cannot be resolved becomes a diagnostic with a character span; the compiler
 * never guesses.
 */
import { argmax } from "@gpu-utils/runtime";
import type { ViewFeatures } from "./features.ts";
import { POLARITY } from "./lexicon.ts";
import {
  type Entry,
  enumEntries,
  fieldEntries,
  forms,
  resolveWords,
  type Schema,
  wordsOf,
} from "./match.ts";
import type { Model } from "./model.ts";
import { resolveTime } from "./time.ts";

export type Role =
  | "O"
  | "FIELD"
  | "OP"
  | "VALUE"
  | "TIME_VALUE"
  | "CONJ"
  | "NEG"
  | "SORT_FIELD"
  | "SORT_DIR"
  | "GROUP_FIELD"
  | "AGG_FN"
  | "AGG_FIELD"
  | "LIMIT"
  | "CHART";
export type FilterOp =
  | "eq"
  | "neq"
  | "contains"
  | "gt"
  | "gte"
  | "lt"
  | "lte"
  | "between"
  | "in"
  | "is_true"
  | "is_false"
  | "is_empty"
  | "not_empty";
export type FilterValue = string | number | (string | number)[];
export type SortDir = "asc" | "desc";
export type AggregateFn = "count" | "sum" | "avg" | "min" | "max";
export type Chart = "table" | "bar" | "line" | "pie" | "number";
export type Granularity = "day" | "week" | "month" | "quarter" | "year";
export type DiagnosticCode =
  | "unknown_field"
  | "unknown_value"
  | "unresolved_value"
  | "ambiguous_value"
  | "invalid_number"
  | "unresolved_time"
  | "no_date_field"
  | "ambiguous_date_field"
  | "incomplete_range"
  | "unsupported_negation"
  | "aggregate_missing_field"
  | "unknown_sort_field"
  | "unknown_group_field"
  | "invalid_limit"
  | "unknown_chart";

export interface Span {
  /** UTF-16 code unit offsets, half-open. */
  start: number;
  end: number;
}
export interface ViewFilter {
  field: string;
  op: FilterOp;
  value?: FilterValue;
  span: Span;
}
export interface ViewSort {
  field: string;
  dir: SortDir;
  span: Span;
}
export interface ViewGroup {
  field: string;
  span: Span;
}
export interface ViewAggregate {
  fn: AggregateFn;
  field?: string;
  span: Span;
}
export interface ViewDiagnostic {
  code: DiagnosticCode;
  message: string;
  span: Span;
}
export interface ViewToken {
  text: string;
  start: number;
  end: number;
  role: Role;
  /** True when this token opens a new clause. */
  boundary: boolean;
}
export interface ViewSpec {
  filters: ViewFilter[];
  sort: ViewSort[];
  groupBy: ViewGroup[];
  aggregate: ViewAggregate[];
  limit?: number;
  chart?: Chart;
  /** Set when a group-by clause names a calendar unit ("per month"). */
  granularity?: Granularity;
  diagnostics: ViewDiagnostic[];
  /** Model tokens with their predicted roles, for debugging and highlighting. */
  tokens: ViewToken[];
}

export interface CompileOptions {
  /** Reference date for relative phrases such as "last 30 days". Defaults to now. */
  now?: Date | string;
  /** Date field used when a time phrase names none and the schema has several date fields. */
  dateField?: string;
}

/** Turns logits into roles/boundaries and compiles them. */
export function decode(
  model: Model,
  features: ViewFeatures,
  logits: Float32Array,
  text: string,
  schema: Schema,
  options: CompileOptions = {},
): ViewSpec {
  const k = model.manifest.labels.length;
  const outs = model.manifest.tags; // roles + the clause-boundary column
  const n = features.tokens.length;
  const roleLogits = new Float32Array(n * k);
  const bounds: boolean[] = [];
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < k; j++) roleLogits[t * k + j] = logits[t * outs + j]!;
    bounds.push(logits[t * outs + k]! > 0);
  }
  const ids = argmax(roleLogits, n, k);
  const roles = Array.from(ids, (i) => (model.manifest.labels[i] ?? "O") as Role);
  return compile(text, features.tokens, roles, bounds, schema, options);
}

interface Tok {
  text: string;
  start: number;
  end: number;
}
interface Run {
  role: Role;
  start: number; // token index, inclusive
  end: number; // token index, exclusive
}
interface Ctx {
  text: string;
  tokens: Tok[];
  roles: Role[];
  schema: Schema;
  fieldEntries: Entry[];
  enumEntries: Entry[];
  now: Date;
  dateField: string | undefined;
  spec: ViewSpec;
  /** A "top N" / "bottom N" seen before its sort field ("top 5 comedy episodes by listens"). */
  pendingDir: SortDir | undefined;
}

const FLIP: Record<FilterOp, FilterOp> = {
  eq: "neq",
  neq: "eq",
  gt: "lte",
  lte: "gt",
  lt: "gte",
  gte: "lt",
  is_true: "is_false",
  is_false: "is_true",
  is_empty: "not_empty",
  not_empty: "is_empty",
  contains: "contains",
  between: "between",
  in: "in",
};
const TRUE_WORDS = new Set(["true", "yes", "on", "enabled", "1", "y", "checked"]);
const FALSE_WORDS = new Set(["false", "no", "off", "disabled", "0", "n", "unchecked"]);
const EMPTY_RE = /\b(without|missing|lacking|empty|blank|null|unset|none|unknown|nothing)\b/;
const UNIT_WORDS: Record<string, Granularity> = {
  day: "day",
  date: "day",
  days: "day",
  daily: "day",
  week: "week",
  weeks: "week",
  weekly: "week",
  month: "month",
  months: "month",
  monthly: "month",
  quarter: "quarter",
  quarters: "quarter",
  quarterly: "quarter",
  year: "year",
  years: "year",
  yearly: "year",
  annually: "year",
  annual: "year",
};
const DATE_DIR_RE = /\b(newest|oldest|latest|earliest|recent)\b/;
const NUMBER_RE = /^([0-9]+(?:\.[0-9]+)?)([a-z%]*)$/;
const MULTIPLIERS: Record<string, number> = { k: 1e3, m: 1e6, b: 1e9, bn: 1e9, mm: 1e6 };

/** "5k" → 5000, "$1,200" → 1200, "2.5m" → 2500000, "30%" → 30, "50 cm" → 50 (unit words ignored). */
export function parseNumber(text: string): number | null {
  const t = text
    .toLowerCase()
    .replace(/[,$€£ ]/g, "")
    .replace(/^(usd|eur|gbp)/, "");
  const m = NUMBER_RE.exec(t);
  if (!m) return null;
  return Number(m[1]) * (MULTIPLIERS[m[2] ?? ""] ?? 1);
}

/** Polarity of a comparative/superlative field word ("cheaper" → low, "tallest" → high), if any. */
function polarityOf(word: string): "high" | "low" | undefined {
  for (const f of forms(word)) {
    // "larger" reduces to "larg"; the silent "e" is restored here, not in the shared stemmer.
    const p = POLARITY[f] ?? POLARITY[`${f}e`];
    if (p) return p;
  }
  return undefined;
}
const isComparative = (w: string) => w.length >= 5 && (w.endsWith("er") || w.endsWith("ier"));
const isSuperlative = (w: string) => w.length >= 6 && (w.endsWith("est") || w.endsWith("iest"));

/** Compiles gold or predicted roles into a spec. Exported so tests can round-trip gold labels. */
export function compile(
  text: string,
  tokens: Tok[],
  roles: Role[],
  bounds: boolean[],
  schema: Schema,
  options: CompileOptions = {},
): ViewSpec {
  const spec: ViewSpec = {
    filters: [],
    sort: [],
    groupBy: [],
    aggregate: [],
    diagnostics: [],
    tokens: tokens.map((t, i) => ({
      text: t.text,
      start: t.start,
      end: t.end,
      role: roles[i] ?? "O",
      boundary: !!bounds[i],
    })),
  };
  const now = options.now === undefined ? new Date() : new Date(options.now);
  const ctx: Ctx = {
    text,
    tokens,
    roles,
    schema,
    fieldEntries: fieldEntries(schema),
    enumEntries: enumEntries(schema),
    now,
    dateField: options.dateField,
    spec,
    pendingDir: undefined,
  };
  // Clause segmentation: a non-O token with the boundary bit (or with no open clause) opens a clause.
  const clauses: number[][] = [];
  let cur: number[] | null = null;
  for (let i = 0; i < tokens.length; i++) {
    const role = roles[i] ?? "O";
    if (role !== "O" && (bounds[i] || cur === null)) {
      cur = [i];
      clauses.push(cur);
    } else if (cur !== null) cur.push(i);
  }
  for (const clause of clauses) {
    compileFilter(ctx, clause);
    compileSort(ctx, clause);
    compileGroup(ctx, clause);
    compileAggregate(ctx, clause);
    compileLimit(ctx, clause);
    compileChart(ctx, clause);
  }
  normalize(ctx);
  return spec;
}

/**
 * Cross-clause clean-up. Clauses are compiled independently, so a phrase whose clause
 * boundaries land inside one logical constraint ("north and east bench", "sorted by height,
 * tallest first") produces redundant or contradictory entries. These rules are pure
 * spec algebra — each one rewrites a spec that could never be what the user meant:
 *
 * - two `eq` filters on the same enum/text field are ANDed and so always empty: they are one
 *   value list (`in`);
 * - `not_empty` on a field that carries another constraint is implied by it;
 * - the same field sorted twice is one sort key, and the later clause carries the direction
 *   the user spelled out ("sorted by height, tallest first");
 * - the same aggregate asked for twice is one aggregate.
 */
function normalize(ctx: Ctx): void {
  const spec = ctx.spec;
  const kindOf = (name: string) => ctx.schema.fields.find((f) => f.name === name)?.kind;

  const merged: ViewFilter[] = [];
  for (const f of spec.filters) {
    const prev = merged[merged.length - 1];
    const listable = kindOf(f.field) === "enum" || kindOf(f.field) === "text";
    if (
      prev &&
      listable &&
      prev.field === f.field &&
      f.op === "eq" &&
      (prev.op === "eq" || prev.op === "in") &&
      typeof f.value === "string"
    ) {
      const values = prev.op === "in" ? (prev.value as (string | number)[]) : [prev.value!];
      merged[merged.length - 1] = {
        field: f.field,
        op: "in",
        value: [...values, f.value],
        span: { start: Math.min(prev.span.start, f.span.start), end: Math.max(prev.span.end, f.span.end) },
      };
      continue;
    }
    merged.push(f);
  }
  spec.filters = merged.filter(
    (f) => f.op !== "not_empty" || !merged.some((g) => g !== f && g.field === f.field),
  );

  const sort: ViewSort[] = [];
  for (const s of spec.sort) {
    const at = sort.findIndex((x) => x.field === s.field);
    if (at < 0) sort.push(s);
    else sort[at] = { ...sort[at]!, dir: s.dir };
  }
  spec.sort = sort;

  const seen = new Set<string>();
  spec.aggregate = spec.aggregate.filter((a) => {
    const key = `${a.fn} ${a.field ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function runs(ctx: Ctx, clause: number[], role: Role): Run[] {
  const out: Run[] = [];
  for (const i of clause) {
    if (ctx.roles[i] !== role) continue;
    const last = out[out.length - 1];
    if (last && last.end === i) last.end = i + 1;
    else out.push({ role, start: i, end: i + 1 });
  }
  return out;
}
const runText = (ctx: Ctx, r: Run) =>
  ctx.text.slice(ctx.tokens[r.start]!.start, ctx.tokens[r.end - 1]!.end);
const runSpan = (ctx: Ctx, r: Run): Span => ({
  start: ctx.tokens[r.start]!.start,
  end: ctx.tokens[r.end - 1]!.end,
});
function clauseSpan(ctx: Ctx, clause: number[]): Span {
  let start = Number.POSITIVE_INFINITY;
  let end = 0;
  for (const i of clause) {
    if (ctx.roles[i] === "O") continue;
    start = Math.min(start, ctx.tokens[i]!.start);
    end = Math.max(end, ctx.tokens[i]!.end);
  }
  return { start, end };
}
function diag(ctx: Ctx, code: DiagnosticCode, message: string, span: Span): undefined {
  ctx.spec.diagnostics.push({ code, message, span });
  return undefined;
}
function defaultDateField(ctx: Ctx, span: Span): number {
  const dates = ctx.schema.fields.map((f, i) => (f.kind === "date" ? i : -1)).filter((i) => i >= 0);
  if (ctx.dateField !== undefined) {
    const i = ctx.schema.fields.findIndex((f) => f.name === ctx.dateField);
    if (i >= 0) return i;
  }
  if (dates.length === 1) return dates[0]!;
  const primary = dates.find((i) => ctx.schema.fields[i]!.primary);
  if (primary !== undefined) return primary;
  if (dates.length === 0)
    diag(ctx, "no_date_field", "a time phrase was used but the schema has no date field", span);
  else
    diag(
      ctx,
      "ambiguous_date_field",
      `the schema has ${dates.length} date fields; pass options.dateField or name the field`,
      span,
    );
  return -1;
}

function compileFilter(ctx: Ctx, clause: number[]): undefined {
  const fieldRuns = runs(ctx, clause, "FIELD");
  const valueRuns = runs(ctx, clause, "VALUE");
  let timeRuns = runs(ctx, clause, "TIME_VALUE");
  let negs = runs(ctx, clause, "NEG").length;
  if (fieldRuns.length === 0 && valueRuns.length === 0 && timeRuns.length === 0) return;
  const span = clauseSpan(ctx, clause);
  const opText = runs(ctx, clause, "OP")
    .map((r) => runText(ctx, r).toLowerCase())
    .join(" ");

  let fi = -1;
  let fieldWord = "";
  if (fieldRuns.length > 0) {
    const run = fieldRuns[0]!;
    const words = wordsOf(runText(ctx, run));
    const m = resolveWords(words, ctx.fieldEntries);
    if (!m) {
      diag(
        ctx,
        "unknown_field",
        `"${runText(ctx, run)}" does not match any field`,
        runSpan(ctx, run),
      );
      return;
    }
    fi = m.field;
    if (m.neg) negs++; // "unarchived repos": the negation is inside the field word
    if (words.length === 1) fieldWord = words[0]!;
    // "business or tech episodes": a weak text-field mention next to values that all belong to
    // one enum field is a carrier noun, not the field being filtered.
    if (ctx.schema.fields[fi]!.kind === "text" && valueRuns.length > 0) {
      let owners: Set<number> | null = null;
      for (const v of valueRuns) {
        const e = resolveWords(wordsOf(runText(ctx, v)), ctx.enumEntries);
        if (!e) {
          owners = null;
          break;
        }
        owners =
          owners === null ? new Set(e.owners) : new Set(e.owners.filter((o) => owners!.has(o)));
      }
      if (owners?.size === 1) fi = [...owners][0]!;
    }
  } else if (valueRuns.length > 0) {
    let owners: Set<number> | null = null;
    for (const run of valueRuns) {
      const m = resolveWords(wordsOf(runText(ctx, run)), ctx.enumEntries);
      if (!m) {
        diag(
          ctx,
          "unresolved_value",
          `"${runText(ctx, run)}" is not a known value of any field`,
          runSpan(ctx, run),
        );
        return;
      }
      owners =
        owners === null ? new Set(m.owners) : new Set(m.owners.filter((o) => owners!.has(o)));
    }
    if (owners?.size !== 1) {
      diag(
        ctx,
        "ambiguous_value",
        `"${runText(ctx, valueRuns[0]!)}" belongs to more than one field`,
        span,
      );
      return;
    }
    fi = [...owners][0]!;
  } else {
    fi = defaultDateField(ctx, span);
    if (fi < 0) return;
  }
  const field = ctx.schema.fields[fi]!;
  const flip = (op: FilterOp): FilterOp => {
    let o = op;
    for (let i = 0; i < negs; i++) o = FLIP[o];
    return o;
  };
  const push = (op: FilterOp, value?: FilterValue): undefined => {
    const f: ViewFilter =
      value === undefined
        ? { field: field.name, op, span }
        : { field: field.name, op, value, span };
    ctx.spec.filters.push(f);
    return undefined;
  };
  const negatedUnsupported = (op: FilterOp) => {
    if (negs % 2 === 1) {
      diag(ctx, "unsupported_negation", `a negated "${op}" filter is not supported`, span);
      return true;
    }
    return false;
  };

  if (field.kind === "date" && timeRuns.length === 0 && valueRuns.length > 0) timeRuns = valueRuns;
  // "vintage between 2015 and 2020": years on a numeric field are numbers.
  const numberRuns =
    field.kind === "number"
      ? [...valueRuns, ...timeRuns].sort((a, b) => a.start - b.start)
      : valueRuns;
  const hasValues = field.kind === "date" ? timeRuns.length > 0 : numberRuns.length > 0;
  if (!hasValues) {
    if (field.kind === "boolean") return push(flip(EMPTY_RE.test(opText) ? "is_false" : "is_true"));
    return push(flip(EMPTY_RE.test(opText) ? "is_empty" : "not_empty"));
  }

  if (field.kind === "boolean") {
    const v = runText(ctx, valueRuns[0]!).toLowerCase();
    if (TRUE_WORDS.has(v)) return push(flip("is_true"));
    if (FALSE_WORDS.has(v)) return push(flip("is_false"));
    // "active customers": the flag still constrains the view; only the stray word is reported.
    diag(ctx, "unknown_value", `"${v}" is not a boolean value`, runSpan(ctx, valueRuns[0]!));
    return push(flip(EMPTY_RE.test(opText) ? "is_false" : "is_true"));
  }

  if (field.kind === "number") {
    const nums: number[] = [];
    for (const run of numberRuns) {
      const v = parseNumber(runText(ctx, run));
      if (v === null)
        return diag(
          ctx,
          "invalid_number",
          `"${runText(ctx, run)}" is not a number`,
          runSpan(ctx, run),
        );
      nums.push(v);
    }
    if (nums.length >= 2 || /\bbetween\b/.test(opText)) {
      if (nums.length < 2) return diag(ctx, "incomplete_range", "a range needs two numbers", span);
      if (negatedUnsupported("between")) return;
      return push("between", [nums[0]!, nums[1]!]);
    }
    return push(flip(numberOp(opText, fieldWord, isYear(nums[0]!))), nums[0]!);
  }

  if (field.kind === "date") {
    const ranges: [string, string][] = [];
    for (const run of timeRuns) {
      const r = resolveTime(runText(ctx, run), ctx.now);
      if (!r)
        return diag(
          ctx,
          "unresolved_time",
          `"${runText(ctx, run)}" is not a recognised date phrase`,
          runSpan(ctx, run),
        );
      ranges.push(r);
    }
    if (ranges.length >= 2 || /\bbetween\b/.test(opText)) {
      if (ranges.length < 2)
        return diag(ctx, "incomplete_range", "a date range needs two dates", span);
      if (negatedUnsupported("between")) return;
      return push("between", [ranges[0]![0], ranges[1]![1]]);
    }
    const [lo, hi] = ranges[0]!;
    let fam = dateOp(opText);
    if (fam === "in") {
      if (lo === hi) return push(flip("eq"), lo);
      if (negatedUnsupported("between")) return;
      return push("between", [lo, hi]);
    }
    fam = flip(fam);
    if (/\bago$/.test(runText(ctx, timeRuns[0]!).trim().toLowerCase()))
      fam =
        ({ lt: "gt", gt: "lt", lte: "gte", gte: "lte" } as Record<string, FilterOp>)[fam] ?? fam;
    return push(fam, fam === "lt" || fam === "gte" ? lo : hi);
  }

  if (field.kind === "enum") {
    const own = ctx.enumEntries.filter((e) => e.field === fi);
    const values: string[] = [];
    for (const run of valueRuns) {
      const m = resolveWords(wordsOf(runText(ctx, run)), own);
      if (!m)
        return diag(
          ctx,
          "unknown_value",
          `"${runText(ctx, run)}" is not a value of ${field.name}`,
          runSpan(ctx, run),
        );
      values.push(field.values![m.value]!);
    }
    if (values.length >= 2) {
      if (negs % 2 === 1) for (const v of values) push("neq", v);
      else push("in", values);
      return;
    }
    return push(flip("eq"), values[0]!);
  }

  // text
  const values = valueRuns.map((r) => runText(ctx, r));
  if (values.length >= 2) {
    if (negs % 2 === 1) for (const v of values) push("neq", v);
    else push("in", values);
    return;
  }
  const v = values[0]!;
  const eqLike =
    /\b(is|equals?|exactly|named|called|titled|set)\b|[=:]/.test(opText) ||
    v.toLowerCase() === "me";
  if (eqLike) return push(flip("eq"), v);
  if (negatedUnsupported("contains")) return;
  push("contains", v);
}

/** A numeric value that reads as a calendar year ("vintage 2019", "published before 1800"). */
const isYear = (v: number) => Number.isInteger(v) && v >= 1500 && v <= 2100;

function numberOp(op: string, fieldWord = "", year = false): FilterOp {
  const has = (re: RegExp) => re.test(op);
  // "2019 or older" on a year-like field is an upper bound, not a lower one.
  if (year) {
    if (has(/\bor (older|earlier)\b/)) return "lte";
    if (has(/\bor (newer|later)\b/)) return "gte";
    if (has(/\b(older|earlier)\b/)) return "lt";
    if (has(/\b(newer|later)\b/)) return "gt";
  }
  // "cheaper than 100" / "taller than 50": the comparative field word carries the direction.
  if (
    isComparative(fieldWord) &&
    !has(/\b(more|less|greater|fewer|at least|at most|up to|minimum|maximum|min|max)\b|[<>≥≤=]/)
  ) {
    const p = polarityOf(fieldWord);
    if (p === "high") return has(/\bor equal\b/) ? "gte" : "gt";
    if (p === "low") return has(/\bor equal\b/) ? "lte" : "lt";
  }
  if (
    has(
      /\bat least\b|\bminimum\b|\bmin\b|>=|≥|\bor more\b|\bor higher\b|\band up\b|\band above\b|\band over\b|\bstarting at\b|\+/,
    )
  )
    return "gte";
  if (
    has(
      /\bat most\b|\bmaximum\b|\bmax\b|<=|≤|\bup to\b|\bor less\b|\bor fewer\b|\bor lower\b|\band under\b|\band below\b|\bcapped\b/,
    )
  )
    return "lte";
  const orEqual = has(/\bor equal\b/);
  if (
    has(
      /\b(more|greater|over|above|exceed(s|ing)?|bigger|higher|larger|beyond|past|longer|taller|heavier|wider|deeper|faster|older)\b|>/,
    )
  )
    return orEqual ? "gte" : "gt";
  if (
    has(
      /\b(less|fewer|under|below|smaller|lower|beneath|shorter|lighter|narrower|slower|cheaper|younger|newer)\b|</,
    )
  )
    return orEqual ? "lte" : "lt";
  return "eq";
}

function dateOp(op: string): FilterOp | "in" {
  const has = (re: RegExp) => re.test(op);
  if (has(/\b(until|till|through|up to|by|or before|at most|no later)\b|<=|≤/)) return "lte";
  if (has(/\b(before|prior|earlier|older|less|fewer|under|below)\b|</)) return "lt";
  if (has(/\b(since|starting|from|or after|at least)\b|>=|≥/)) return "gte";
  if (has(/\b(after|later|newer|past|more|greater|over|above)\b|>/)) return "gt";
  return "in";
}

function compileSort(ctx: Ctx, clause: number[]): void {
  const fieldRuns = runs(ctx, clause, "SORT_FIELD");
  const dirRuns = runs(ctx, clause, "SORT_DIR");
  if (fieldRuns.length === 0 && dirRuns.length === 0) return;
  const span = clauseSpan(ctx, clause);
  const dirText = dirRuns.map((r) => runText(ctx, r).toLowerCase()).join(" ");
  if (fieldRuns.length === 0) {
    if (DATE_DIR_RE.test(dirText)) {
      const fi = defaultDateField(ctx, span);
      if (fi >= 0)
        ctx.spec.sort.push({
          field: ctx.schema.fields[fi]!.name,
          dir: sortDir(dirText) ?? "asc",
          span,
        });
    } else if (/\b(top|bottom)\b/.test(dirText)) {
      ctx.pendingDir = /\bbottom\b/.test(dirText) ? "asc" : "desc"; // attaches to the next "by field"
    }
    return;
  }
  for (const run of fieldRuns) {
    const words = wordsOf(runText(ctx, run));
    const m = resolveWords(words, ctx.fieldEntries);
    if (!m) {
      diag(
        ctx,
        "unknown_sort_field",
        `cannot sort by "${runText(ctx, run)}": no such field`,
        runSpan(ctx, run),
      );
      continue;
    }
    let dir = sortDir(dirText);
    if (dir === undefined && words.length === 1 && isSuperlative(words[0]!)) {
      const p = polarityOf(words[0]!); // "cheapest" → asc, "tallest" → desc
      if (p) dir = p === "high" ? "desc" : "asc";
    }
    if (dir === undefined && dirRuns.length === 0 && ctx.pendingDir !== undefined) {
      dir = ctx.pendingDir;
      ctx.pendingDir = undefined;
    }
    ctx.spec.sort.push({ field: ctx.schema.fields[m.field]!.name, dir: dir ?? "asc", span });
  }
}

/** Explicit direction words; undefined when the text carries none ("first", ""). */
function sortDir(t: string): SortDir | undefined {
  if (/low to high|lowest to highest|a to z|increasing/.test(t)) return "asc";
  if (/high to low|highest to lowest|z to a|decreasing/.test(t)) return "desc";
  if (
    /\b(desc|descending|highest|top|newest|latest|recent|largest|biggest|reverse|reversed|most|best|longest|tallest|heaviest|expensive|priciest)\b/.test(
      t,
    )
  )
    return "desc";
  if (
    /\b(asc|ascending|lowest|bottom|oldest|earliest|alphabetical|alphabetically|smallest|least|worst|fewest|cheapest|shortest|lightest)\b/.test(
      t,
    )
  )
    return "asc";
  return undefined;
}

function compileGroup(ctx: Ctx, clause: number[]): void {
  for (const run of runs(ctx, clause, "GROUP_FIELD")) {
    const words = wordsOf(runText(ctx, run));
    const span = runSpan(ctx, run);
    if (words.length === 1 && UNIT_WORDS[words[0]!]) {
      const fi = defaultDateField(ctx, span);
      if (fi < 0) continue;
      ctx.spec.groupBy.push({ field: ctx.schema.fields[fi]!.name, span });
      ctx.spec.granularity = UNIT_WORDS[words[0]!]!;
      continue;
    }
    const m = resolveWords(words, ctx.fieldEntries);
    if (!m) {
      diag(
        ctx,
        "unknown_group_field",
        `cannot group by "${runText(ctx, run)}": no such field`,
        span,
      );
      continue;
    }
    ctx.spec.groupBy.push({ field: ctx.schema.fields[m.field]!.name, span });
    const rest = words.slice(m.end);
    const unit = rest.length === 1 ? UNIT_WORDS[rest[0]!] : undefined;
    if (unit && ctx.schema.fields[m.field]!.kind === "date") ctx.spec.granularity = unit;
  }
}

function aggregateFn(t: string): AggregateFn {
  if (/\b(count|number|many|amount)\b|#/.test(t)) return "count";
  if (/\b(avg|average|mean|median)\b/.test(t)) return "avg";
  if (/\b(min|minimum|lowest|smallest|least|fewest)\b/.test(t)) return "min";
  if (/\b(max|maximum|highest|largest|biggest|peak|most)\b/.test(t)) return "max";
  return "sum";
}

function compileAggregate(ctx: Ctx, clause: number[]): void {
  const fnRuns = runs(ctx, clause, "AGG_FN");
  const fieldRuns = runs(ctx, clause, "AGG_FIELD");
  if (fnRuns.length === 0) return;
  const span = clauseSpan(ctx, clause);
  const items = fnRuns.map((r) => ({
    fn: aggregateFn(runText(ctx, r).toLowerCase()),
    at: r.start,
    field: undefined as string | undefined,
    bad: false,
  }));
  for (const run of fieldRuns) {
    const m = resolveWords(wordsOf(runText(ctx, run)), ctx.fieldEntries);
    if (!m) {
      diag(
        ctx,
        "unknown_field",
        `"${runText(ctx, run)}" does not match any field`,
        runSpan(ctx, run),
      );
      continue;
    }
    const before = items.filter((it) => it.at < run.start && it.field === undefined);
    const target =
      before[before.length - 1] ?? items.find((it) => it.at > run.start && it.field === undefined);
    if (target) target.field = ctx.schema.fields[m.field]!.name;
  }
  for (const it of items) {
    if (it.fn !== "count" && it.field === undefined) {
      diag(ctx, "aggregate_missing_field", `"${it.fn}" needs a numeric field`, span);
      continue;
    }
    ctx.spec.aggregate.push(
      it.field === undefined ? { fn: it.fn, span } : { fn: it.fn, field: it.field, span },
    );
  }
}

function compileLimit(ctx: Ctx, clause: number[]): void {
  const run = runs(ctx, clause, "LIMIT")[0];
  if (!run) return;
  const v = parseNumber(runText(ctx, run));
  if (v === null || !Number.isInteger(v) || v <= 0) {
    diag(ctx, "invalid_limit", `"${runText(ctx, run)}" is not a valid limit`, runSpan(ctx, run));
    return;
  }
  if (ctx.spec.limit === undefined) ctx.spec.limit = v;
}

function compileChart(ctx: Ctx, clause: number[]): void {
  const chartRuns = runs(ctx, clause, "CHART");
  if (chartRuns.length === 0) return;
  const t = chartRuns.map((r) => runText(ctx, r).toLowerCase()).join(" ");
  let chart: Chart | undefined;
  if (/\b(bar|bars|column|columns|histogram)\b/.test(t)) chart = "bar";
  else if (/\b(line|lines|trend|timeseries)\b/.test(t)) chart = "line";
  else if (/\b(pie|donut|doughnut)\b/.test(t)) chart = "pie";
  else if (/\b(table|tabular|grid|list|rows)\b/.test(t)) chart = "table";
  else if (/\b(number|kpi|scorecard|stat|metric|tile|figure|value|single|big)\b/.test(t))
    chart = "number";
  if (chart === undefined) {
    diag(ctx, "unknown_chart", `"${t}" is not a known chart type`, runSpan(ctx, chartRuns[0]!));
    return;
  }
  if (ctx.spec.chart === undefined) ctx.spec.chart = chart;
}
