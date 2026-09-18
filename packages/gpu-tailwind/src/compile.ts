/**
 * Deterministic compiler: typed values, (property, value) -> Tailwind v4 classes,
 * variant resolution and class validation. Mirrors training/gpu_tailwind/semantics.py
 * and lexicon.py; every rule here has a line-for-line twin there (test/oracle.test.ts).
 */
import table from "./table.json" with { type: "json" };

interface Table {
  theme: Record<string, string[]>;
  props: Record<string, string>;
  vals: Record<string, string>;
  mods: Record<string, string>;
  vars: {
    tiers: Record<string, string[]>;
    smLiteral: string[];
    max: string[];
    min: string[];
    xlWords: string[];
    lgFallback: string[];
    only: string[];
    strong: [string, string[], string[]][];
    weak: [string, string[], string[]][];
  };
  presets: Record<string, string>;
  seps: string[];
  neg: string[];
  spelling: Record<string, string>;
}
export const T = table as unknown as Table;

export type Kind =
  | "sz"
  | "kw"
  | "col"
  | "mod"
  | "wt"
  | "rad"
  | "num"
  | "frac"
  | "pct"
  | "unit"
  | "spec"
  | "lit"
  | "int";
export interface Value {
  kind: Kind;
  value: string;
  intensity: number;
}

export const HUES = T.theme.hues!;
export const SHADES = T.theme.shades!;
const SIZES = [
  "xs",
  "sm",
  "md",
  "lg",
  "xl",
  "2xl",
  "3xl",
  "4xl",
  "5xl",
  "6xl",
  "7xl",
  "8xl",
  "9xl",
];
const WEIGHTS = [
  "thin",
  "extralight",
  "light",
  "normal",
  "medium",
  "semibold",
  "bold",
  "extrabold",
  "black",
];
const SPACING = new Set([
  "p",
  "px",
  "py",
  "pt",
  "pr",
  "pb",
  "pl",
  "m",
  "mx",
  "my",
  "mt",
  "mr",
  "mb",
  "ml",
  "gap",
  "gap-x",
  "gap-y",
  "space-x",
  "space-y",
]);
const INSETS = new Set(["top", "bottom", "left", "right", "inset"]);
const SIZING = new Set(["w", "h", "size", "min-w", "max-w", "min-h", "max-h"]);
const SZ_SPACING: Record<string, string> = {
  none: "0",
  xs: "1",
  sm: "2",
  md: "4",
  lg: "8",
  xl: "12",
  "2xl": "16",
  "3xl": "24",
  "4xl": "32",
  "5xl": "40",
  "6xl": "48",
  "7xl": "64",
};
const SZ_HEIGHT: Record<string, string> = {
  none: "0",
  xs: "8",
  sm: "12",
  md: "16",
  lg: "24",
  xl: "32",
  "2xl": "48",
  "3xl": "64",
  "4xl": "96",
};
const SZ_DURATION: Record<string, string> = {
  xs: "75",
  sm: "150",
  md: "300",
  lg: "500",
  xl: "700",
  "2xl": "1000",
};
const SZ_OPACITY: Record<string, string> = {
  none: "0",
  xs: "10",
  sm: "25",
  md: "50",
  lg: "75",
  xl: "90",
  full: "100",
};
const WEIGHT_NUM: Record<string, string> = {
  "100": "thin",
  "200": "extralight",
  "300": "light",
  "400": "normal",
  "500": "medium",
  "600": "semibold",
  "700": "bold",
  "800": "extrabold",
  "900": "black",
};
const LENGTH_UNITS = ["px", "rem", "em", "vh", "vw", "ch"];

const V = (kind: Kind, value: string, intensity = 0): Value => ({ kind, value, intensity });
const parseKV = (s: string): Value => {
  const i = s.indexOf(":");
  return V(s.slice(0, i) as Kind, s.slice(i + 1));
};

// ---------------------------------------------------------------- words

export function normalizeWord(w: string): string {
  const l = w.toLowerCase();
  if (l === "percent" || l === "pct") return "%";
  return T.spelling[l] ?? l;
}

/** Lowercase words; every punctuation character is its own word (mirror of words_of). */
export function wordsOf(text: string): string[] {
  const m = text.toLowerCase().match(/[a-z0-9]+|[^\sa-z0-9]/g);
  return (m ?? []).map(normalizeWord);
}

let vocab: Set<string> | undefined;
function vocabulary(): Set<string> {
  if (vocab) return vocab;
  vocab = new Set<string>();
  const add = (s: string) => {
    for (const w of s.split(" ")) if (w.length >= 3) vocab!.add(w);
  };
  for (const k of Object.keys(T.props)) add(k);
  for (const k of Object.keys(T.vals)) add(k);
  for (const k of Object.keys(T.mods)) add(k);
  for (const h of HUES) vocab.add(h);
  for (const list of Object.values(T.vars.tiers)) for (const w of list) vocab.add(w);
  for (const rule of [...T.vars.strong, ...T.vars.weak])
    for (const w of [...rule[1], ...rule[2]]) vocab.add(w);
  for (const w of [...T.vars.max, ...T.vars.min, ...T.vars.only]) vocab.add(w);
  return vocab;
}

/** Damerau-Levenshtein distance <= 1 check. */
function within1(a: string, b: string): boolean {
  if (a === b) return true;
  const la = a.length;
  const lb = b.length;
  if (Math.abs(la - lb) > 1) return false;
  if (la === lb) {
    let diff = -1;
    for (let i = 0; i < la; i++) {
      if (a[i] !== b[i]) {
        if (diff >= 0) {
          // transposition?
          return (
            diff === i - 1 &&
            a[i] === b[diff] &&
            a[diff] === b[i] &&
            a.slice(i + 1) === b.slice(i + 1)
          );
        }
        diff = i;
      }
    }
    return true;
  }
  const [s, l] = la < lb ? [a, b] : [b, a];
  let i = 0;
  while (i < s.length && s[i] === l[i]) i++;
  return s.slice(i) === l.slice(i + 1);
}

/** Vocabulary words at Damerau-Levenshtein distance 1 from an unknown word (>= 4 chars). */
function candidates(w: string): string[] {
  const voc = vocabulary();
  if (w.length < 4 || voc.has(w) || /\d/.test(w)) return [];
  const out: string[] = [];
  for (const c of voc)
    if (c.length >= 5 && Math.abs(c.length - w.length) <= 1 && within1(w, c)) out.push(c);
  return out;
}

/**
 * Alternative spellings of a phrase with each unknown word replaced by a vocabulary word at
 * edit distance 1 (all combinations, capped). Empty when nothing needs correcting.
 */
export function correctWordsAll(words: string[], cap = 16): string[][] {
  let variants: string[][] = [[]];
  let changed = false;
  for (const w of words) {
    const cs = candidates(w);
    if (cs.length === 0) {
      variants = variants.map((v) => [...v, w]);
      continue;
    }
    changed = true;
    const next: string[][] = [];
    for (const v of variants) for (const c of cs) if (next.length < cap) next.push([...v, c]);
    variants = next;
  }
  return changed ? variants : [];
}

/** First correction candidate (kept for callers that only need one). */
export function correctWords(words: string[]): string[] | null {
  return correctWordsAll(words)[0] ?? null;
}

// ---------------------------------------------------------------- values

export function shiftSize(sz: string, by: number, lo = "xs", hi = "3xl"): string {
  if (!SIZES.includes(sz) || by === 0) return sz;
  const i = Math.max(SIZES.indexOf(lo), Math.min(SIZES.indexOf(hi), SIZES.indexOf(sz) + by));
  return SIZES[i]!;
}
const sizeLe = (s: string, hi: string) =>
  SIZES.includes(s) && SIZES.indexOf(s) <= SIZES.indexOf(hi);
export function shiftShade(shade: string, by: number): string {
  const i = Math.max(0, Math.min(SHADES.length - 1, SHADES.indexOf(shade) + 2 * by));
  return SHADES[i]!;
}
function shiftWeight(w: string, by: number): string {
  if (!WEIGHTS.includes(w) || by === 0) return w;
  return WEIGHTS[Math.max(0, Math.min(WEIGHTS.length - 1, WEIGHTS.indexOf(w) + by))]!;
}

const COLOR_FILLER = new Set([
  "a",
  "an",
  "the",
  "shade",
  "shades",
  "of",
  "in",
  "at",
  "but",
  "tone",
  "tint",
  "version",
  "variant",
  "color",
  "colour",
  "coloured",
  "colored",
  "with",
  "opacity",
  "alpha",
  "transparency",
]);
const ALPHA_WORDS: Record<string, string> = {
  translucent: "50",
  "semi transparent": "50",
  "semi-transparent": "50",
  "see through": "50",
  "half transparent": "50",
  "mostly transparent": "25",
  "slightly transparent": "75",
  "barely transparent": "90",
  "very transparent": "25",
  faintly: "25",
};

/** [alpha words] [modifiers] hue [shade] | hue-shade | shade hue, with "at N%" style alpha. */
function parseColor(words: string[]): Value | null {
  if (words.length === 0) return null;
  let ws = [...words];
  let alpha: string | undefined;
  // alpha: "N %" anywhere, or "/ N" at the end
  for (let i = 0; i + 1 < ws.length; i++) {
    if (/^\d+$/.test(ws[i]!) && ws[i + 1] === "%") {
      alpha = ws[i];
      ws.splice(i, 2);
      break;
    }
  }
  if (
    alpha === undefined &&
    ws.length >= 3 &&
    ws[ws.length - 2] === "/" &&
    /^\d+$/.test(ws[ws.length - 1]!)
  ) {
    alpha = ws[ws.length - 1];
    ws = ws.slice(0, -2);
  }
  for (const [phrase, a] of Object.entries(ALPHA_WORDS)) {
    const pw = phrase.split(" ");
    if (ws.length > pw.length && pw.every((w, i) => ws[i] === w)) {
      alpha = alpha ?? a;
      ws = ws.slice(pw.length);
      break;
    }
  }
  ws = ws.filter((w) => !COLOR_FILLER.has(w));
  if (ws.length === 0) return null;
  const joined = ws.join("");
  const m = /^([a-z]+)-?(\d{2,3})$/.exec(joined);
  const withAlpha = (v: string) => V("col", alpha !== undefined ? `${v}/${alpha}` : v);
  if (m && HUES.includes(m[1]!) && SHADES.includes(m[2]!)) return withAlpha(`${m[1]}-${m[2]}`);
  let shade: string | undefined;
  if (ws.length && SHADES.includes(ws[ws.length - 1]!)) shade = ws.pop();
  else if (ws.length && SHADES.includes(ws[0]!)) shade = ws.shift();
  if (ws.length === 0) return null;
  // "blue but darker": a modifier after the hue
  let hueIndex = ws.findIndex((w) => HUES.includes(w));
  if (hueIndex < 0) {
    if (ws.length === 1 && (ws[0] === "white" || ws[0] === "black" || ws[0] === "transparent"))
      return withAlpha(ws[0]!);
    return null;
  }
  const hue = ws[hueIndex]!;
  const mods = [...ws.slice(0, hueIndex), ...ws.slice(hueIndex + 1)];
  hueIndex = 0;
  if (mods.length) {
    const mod = T.mods[mods.join(" ")];
    if (mod === undefined) return null;
    shade = shade ?? mod;
  }
  return withAlpha(`${hue}-${shade ?? "500"}`);
}

/** Mirror of parse_value in semantics.py. */
export function parseValue(text: string): Value | null {
  const words = wordsOf(text);
  if (words.length === 0) return null;
  const joined = words.join("");
  const phrase = words.join(" ");
  if (T.vals[phrase] !== undefined) return parseKV(T.vals[phrase]!);
  if (T.mods[phrase] !== undefined) return V("mod", T.mods[phrase]!);
  let intensity = 0;
  let rest = words;
  for (;;) {
    if (rest.length === 0) break;
    const two = rest.slice(0, 2).join(" ");
    const one = rest[0]!;
    if (T.vals[two]?.startsWith("int:")) {
      intensity += Number(T.vals[two]!.slice(4));
      rest = rest.slice(2);
    } else if (T.vals[one]?.startsWith("int:")) {
      intensity += Number(T.vals[one]!.slice(4));
      rest = rest.slice(1);
    } else break;
  }
  if (intensity !== 0 && rest.length) {
    const inner = parseValue(rest.join(" "));
    if (inner) {
      inner.intensity += intensity;
      return inner;
    }
    return null;
  }
  if (T.vals[joined] !== undefined) return parseKV(T.vals[joined]!);
  const col = parseColor(words);
  if (col) return col;
  if (/^\d+(\.\d+)?$/.test(joined)) return V("num", joined);
  if (/^\d+(\.\d+)?(px|rem|em|vh|vw|ch|ms|s)$/.test(joined)) return V("unit", joined);
  if (/^\d+%$/.test(joined)) return V("pct", joined.slice(0, -1));
  if (/^\d+\/\d+$/.test(joined)) return V("frac", joined);
  if (
    words.length === 2 &&
    ["light", "dark", "pale", "deep"].includes(words[0]!) &&
    (words[1] === "gray" || words[1] === "grey")
  ) {
    return V("col", `gray-${T.mods[words[0]!]}`);
  }
  return null;
}

// ---------------------------------------------------------------- emission

const isInt = (x: string, lo: number, hi: number) =>
  /^\d+$/.test(x) && Number(x) >= lo && Number(x) <= hi;
const isSpacingNum = (x: string) =>
  /^\d+$/.test(x) ? Number(x) <= 96 : /^\d+\.5$/.test(x) && Number(x) <= 12;
const unitOk = (v: Value, units: string[]) =>
  v.kind === "unit" && units.some((u) => v.value.endsWith(u));

function spacingValue(k: string, v: Value | null, neg: boolean): string[] {
  if (neg) return [`${k}-0`];
  if (!v) return INSETS.has(k) ? [`${k}-0`] : [`${k}-4`];
  if (v.kind === "num") return isSpacingNum(v.value) ? [`${k}-${v.value}`] : [];
  if (v.kind === "unit") return unitOk(v, LENGTH_UNITS) ? [`${k}-[${v.value}]`] : [];
  if (v.kind === "pct") return [`${k}-[${v.value}%]`];
  if (v.kind === "frac") return INSETS.has(k) ? [`${k}-${v.value}`] : [];
  if (v.kind === "sz") {
    if (v.value === "full") return INSETS.has(k) ? [`${k}-full`] : [];
    const s = v.value !== "none" ? shiftSize(v.value, v.intensity, "xs", "7xl") : "none";
    const n = SZ_SPACING[s];
    return n ? [`${k}-${n}`] : [];
  }
  if (v.kind === "kw") {
    if (v.value === "auto" && (k.startsWith("m") || INSETS.has(k))) return [`${k}-auto`];
    if (v.value === "negative") return [`-${k}-4`];
    if (["top", "bottom", "left", "right"].includes(v.value) && INSETS.has(k)) return [`${k}-0`];
  }
  return [];
}

function sizingValue(k: string, v: Value | null, neg: boolean): string[] {
  if (neg) return [`${k}-0`];
  if (!v) {
    const d: Record<string, string[]> = {
      w: ["w-full"],
      h: ["h-full"],
      size: ["size-full"],
      "min-w": ["min-w-0"],
      "max-w": ["max-w-full"],
      "min-h": ["min-h-full"],
      "max-h": ["max-h-full"],
    };
    return d[k]!;
  }
  const horizontal = ["w", "min-w", "max-w", "size"].includes(k);
  if (v.kind === "num") return isSpacingNum(v.value) ? [`${k}-${v.value}`] : [];
  if (v.kind === "unit") return unitOk(v, LENGTH_UNITS) ? [`${k}-[${v.value}]`] : [];
  if (v.kind === "pct") return [`${k}-[${v.value}%]`];
  if (v.kind === "frac") return [`${k}-${v.value}`];
  if (v.kind === "sz") {
    if (v.value === "full") return [`${k}-full`];
    if (v.value === "none") return k === "max-w" ? ["max-w-none"] : [`${k}-0`];
    const s = shiftSize(v.value, v.intensity, "xs", "7xl");
    if (!sizeLe(s, "7xl")) return [];
    if (horizontal) return [`${k}-${s}`];
    const n = SZ_HEIGHT[s];
    return n ? [`${k}-${n}`] : [];
  }
  if (v.kind === "kw") {
    if (v.value === "screen") return [`${k}-screen`];
    if (["auto", "fit", "min", "max"].includes(v.value)) return [`${k}-${v.value}`];
    if (v.value === "prose") return ["max-w-prose"];
  }
  if (v.kind === "spec" && (v.value === "full-width" || v.value === "full-height"))
    return [`${k}-full`];
  return [];
}

function colorClass(prefix: string, v: Value): string[] {
  if (v.kind === "col") return [`${prefix}-${v.value}`];
  if (v.kind === "mod") return [`${prefix}-gray-${shiftShade(v.value, v.intensity)}`];
  return [];
}

const TEXT_KW: Record<string, string> = {
  center: "text-center",
  left: "text-left",
  right: "text-right",
  justify: "text-justify",
  start: "text-start",
  end: "text-end",
  uppercase: "uppercase",
  lowercase: "lowercase",
  capitalize: "capitalize",
  "normal-case": "normal-case",
  italic: "italic",
  "not-italic": "not-italic",
  underline: "underline",
  "no-underline": "no-underline",
  "line-through": "line-through",
  overline: "overline",
  truncate: "truncate",
  nowrap: "text-nowrap",
  wrap: "text-wrap",
  balance: "text-balance",
  pretty: "text-pretty",
  "break-words": "break-words",
  "break-all": "break-all",
  mono: "font-mono",
  serif: "font-serif",
  sans: "font-sans",
  tighter: "tracking-tighter",
  tight: "tracking-tight",
  wide: "tracking-wide",
  wider: "tracking-wider",
  widest: "tracking-widest",
  snug: "leading-snug",
  relaxed: "leading-relaxed",
  loose: "leading-loose",
  antialiased: "antialiased",
  pre: "whitespace-pre",
  "pre-wrap": "whitespace-pre-wrap",
  "pre-line": "whitespace-pre-line",
  hidden: "hidden",
  "select-none": "select-none",
  "select-all": "select-all",
  "select-text": "select-text",
  invisible: "invisible",
  grayscale: "grayscale",
};
const NEG_KW: Record<string, string> = {
  uppercase: "normal-case",
  lowercase: "normal-case",
  capitalize: "normal-case",
  italic: "not-italic",
  underline: "no-underline",
  wrap: "text-nowrap",
  nowrap: "text-wrap",
  truncate: "text-wrap",
};

function kwText(kw: string, neg: boolean): string[] {
  if (neg && NEG_KW[kw]) return [NEG_KW[kw]!];
  const c = TEXT_KW[kw];
  return c ? [c] : [];
}

function textValue(v: Value | null, neg: boolean): string[] {
  if (!v) return [];
  if (v.kind === "sz") {
    if (v.value === "none" || v.value === "full") return [];
    const s = shiftSize(v.value, v.intensity, "xs", "9xl");
    return [`text-${s === "md" ? "base" : s}`];
  }
  if (v.kind === "col") return [`text-${v.value}`];
  if (v.kind === "mod")
    return v.value === "300" ? ["font-light"] : [`text-gray-${shiftShade(v.value, v.intensity)}`];
  if (v.kind === "wt") return neg ? ["font-normal"] : [`font-${shiftWeight(v.value, v.intensity)}`];
  if (v.kind === "num") return isInt(v.value, 6, 200) ? [`text-[${v.value}px]`] : [];
  if (v.kind === "unit") return unitOk(v, ["px", "rem", "em"]) ? [`text-[${v.value}]`] : [];
  if (v.kind === "spec") {
    const d: Record<string, string[]> = {
      center: ["text-center"],
      hcenter: ["text-center"],
      vcenter: ["align-middle"],
      "sr-only": ["sr-only"],
    };
    return d[v.value] ?? [];
  }
  if (v.kind === "kw") return kwText(v.value, neg);
  return [];
}

const STANDALONE_KW: Record<string, string> = {
  center: "flex items-center justify-center",
  between: "flex justify-between",
  around: "flex justify-around",
  evenly: "flex justify-evenly",
  stretch: "items-stretch",
  baseline: "items-baseline",
  row: "flex flex-row",
  col: "flex flex-col",
  "row-reverse": "flex flex-row-reverse",
  "col-reverse": "flex flex-col-reverse",
  reverse: "flex-row-reverse",
  wrap: "flex-wrap",
  nowrap: "whitespace-nowrap",
  "wrap-reverse": "flex-wrap-reverse",
  visible: "block",
  block: "block",
  inline: "inline",
  "inline-block": "inline-block",
  "inline-flex": "inline-flex",
  "inline-grid": "inline-grid",
  contents: "contents",
  table: "table",
  absolute: "absolute",
  relative: "relative",
  fixed: "fixed",
  sticky: "sticky",
  static: "static",
  scroll: "overflow-scroll",
  "overflow-auto": "overflow-auto",
  clip: "overflow-hidden",
  "overflow-visible": "overflow-visible",
  pointer: "cursor-pointer",
  "not-allowed": "cursor-not-allowed",
  wait: "cursor-wait",
  grab: "cursor-grab",
  move: "cursor-move",
  "text-cursor": "cursor-text",
  "default-cursor": "cursor-default",
  "events-none": "pointer-events-none",
  "events-auto": "pointer-events-auto",
  cover: "object-cover",
  contain: "object-contain",
  fill: "object-fill",
  square: "aspect-square",
  video: "aspect-video",
  "aspect-auto": "aspect-auto",
  spin: "animate-spin",
  ping: "animate-ping",
  pulse: "animate-pulse",
  bounce: "animate-bounce",
  disc: "list-disc",
  decimal: "list-decimal",
  first: "order-first",
  last: "order-last",
  colors: "transition-colors",
  all: "transition-all",
  opacity: "transition-opacity",
  shadow: "transition-shadow",
  transform: "transition-transform",
  linear: "ease-linear",
  "ease-in": "ease-in",
  "ease-out": "ease-out",
  "ease-in-out": "ease-in-out",
  fast: "duration-150",
  slow: "duration-500",
  grow: "grow",
  "no-grow": "grow-0",
  shrink: "shrink",
  "no-shrink": "shrink-0",
  "flex-none": "flex-none",
  "flex-auto": "flex-auto",
  blur: "blur-sm",
  isolate: "isolate",
  group: "group",
  peer: "peer",
  "mx-auto": "mx-auto",
  screen: "h-screen",
  fit: "w-fit",
  min: "w-min",
  max: "w-max",
  prose: "max-w-prose",
  top: "top-0",
  bottom: "bottom-0",
};
const STANDALONE_SPEC: Record<string, string> = {
  center: "flex items-center justify-center",
  vcenter: "flex items-center",
  hcenter: "flex justify-center",
  fullscreen: "w-screen h-screen",
  "cover-parent": "absolute inset-0",
  "pin-top": "top-0",
  "pin-bottom": "bottom-0",
  "pin-left": "left-0",
  "pin-right": "right-0",
  "top-right": "top-0 right-0",
  "top-left": "top-0 left-0",
  "bottom-right": "bottom-0 right-0",
  "bottom-left": "bottom-0 left-0",
  "full-width": "w-full",
  "full-height": "h-full",
  "flex-center": "flex items-center justify-center",
  "sr-only": "sr-only",
};
const NEG_STANDALONE: Record<string, string> = {
  visible: "hidden",
  hidden: "block",
  grow: "grow-0",
  shrink: "shrink-0",
  scroll: "overflow-hidden",
  pointer: "cursor-default",
  wrap: "flex-nowrap",
  nowrap: "flex-wrap",
  blur: "blur-none",
  center: "",
  italic: "not-italic",
  underline: "no-underline",
  uppercase: "normal-case",
};

const split = (s: string) => (s ? s.split(" ") : []);

/** Classes for a VAL with no compatible PROP in its segment (mirror of standalone). */
export function standalone(v: Value, neg: boolean): string[] {
  if (v.kind === "col") return neg ? ["bg-transparent"] : [`bg-${v.value}`];
  if (v.kind === "mod") {
    const s = shiftShade(v.value, v.intensity);
    return [`bg-gray-${s}`, ...(Number(s) >= 700 ? ["text-white"] : [])];
  }
  if (v.kind === "wt") return neg ? ["font-normal"] : [`font-${shiftWeight(v.value, v.intensity)}`];
  if (v.kind === "rad") return neg ? ["rounded-none"] : [`rounded-${v.value}`];
  if (v.kind === "sz") {
    if (v.value === "full") return ["w-full"];
    if (v.value === "none") return [];
    const s = shiftSize(v.value, v.intensity, "xs", "9xl");
    return [`text-${s === "md" ? "base" : s}`];
  }
  if (v.kind === "frac") return [`w-${v.value}`];
  if (v.kind === "pct") return [`w-[${v.value}%]`];
  if (v.kind === "spec") return split(STANDALONE_SPEC[v.value] ?? "");
  if (v.kind === "lit") return [v.value];
  if (v.kind === "kw") {
    if (neg) {
      if (v.value in NEG_STANDALONE) return split(NEG_STANDALONE[v.value]!);
      if (TEXT_KW[v.value] && NEG_KW[v.value]) return [NEG_KW[v.value]!];
    }
    if (STANDALONE_KW[v.value]) return split(STANDALONE_KW[v.value]!);
    const c = TEXT_KW[v.value];
    return c ? [c] : [];
  }
  return [];
}

function borderValue(k: string, v: Value | null, neg: boolean): string[] {
  if (neg) return [`${k}-0`];
  if (!v) return [k];
  if (v.kind === "col") return [k, `${k}-${v.value}`];
  if (v.kind === "mod") return [k, `${k}-gray-${shiftShade(v.value, v.intensity)}`];
  if (v.kind === "num")
    return ["0", "1", "2", "4", "8"].includes(v.value)
      ? v.value === "1"
        ? [k]
        : [`${k}-${v.value}`]
      : [];
  if (v.kind === "unit") return unitOk(v, ["px"]) ? [`${k}-[${v.value}]`] : [];
  if (v.kind === "sz") {
    const s = SIZES.includes(v.value) ? shiftSize(v.value, v.intensity, "xs", "xl") : v.value;
    const d: Record<string, string[]> = {
      none: [`${k}-0`],
      xs: [k],
      sm: [k],
      md: [`${k}-2`],
      lg: [`${k}-4`],
      xl: [`${k}-8`],
      full: [`${k}-8`],
    };
    return d[s] ?? [k];
  }
  if (v.kind === "kw" && ["dashed", "dotted", "solid", "double", "none"].includes(v.value))
    return v.value === "none" ? [`${k}-0`] : [k, `border-${v.value}`];
  return [];
}

function roundedValue(k: string, v: Value | null, neg: boolean): string[] {
  if (neg) return [`${k}-none`];
  if (!v) return [`${k}-lg`];
  if (v.kind === "rad") return [`${k}-${v.value}`];
  if (v.kind === "sz") {
    if (v.value === "full") return [`${k}-full`];
    if (v.value === "none") return [`${k}-none`];
    const s = shiftSize(v.value, v.intensity, "xs", "4xl");
    return sizeLe(s, "4xl") ? [`${k}-${s}`] : [];
  }
  if (v.kind === "num") return isInt(v.value, 0, 64) ? [`${k}-[${v.value}px]`] : [];
  if (v.kind === "unit") return unitOk(v, ["px", "rem", "em"]) ? [`${k}-[${v.value}]`] : [];
  if (v.kind === "kw" && v.value === "square") return [`${k}-none`];
  return [];
}

function ringValue(k: string, v: Value | null, neg: boolean): string[] {
  const base = k === "ring" ? "ring-2" : "outline-2";
  const off = k === "ring" ? "ring-0" : "outline-hidden";
  if (neg) return [off];
  if (!v) return [base];
  if (v.kind === "col") return [base, `${k}-${v.value}`];
  if (v.kind === "mod") return [base, `${k}-gray-${shiftShade(v.value, v.intensity)}`];
  if (v.kind === "num")
    return ["0", "1", "2", "4", "8"].includes(v.value) ? [`${k}-${v.value}`] : [];
  if (v.kind === "sz") {
    const s = SIZES.includes(v.value) ? shiftSize(v.value, v.intensity, "xs", "xl") : v.value;
    const d: Record<string, string[]> = {
      none: [off],
      xs: [`${k}-1`],
      sm: [`${k}-1`],
      md: [`${k}-2`],
      lg: [`${k}-4`],
      xl: [`${k}-8`],
      full: [`${k}-8`],
    };
    return d[s] ?? [base];
  }
  if (v.kind === "kw" && (v.value === "hidden" || v.value === "none")) return [off];
  return [];
}

function shadowValue(v: Value | null, neg: boolean): string[] {
  if (neg) return ["shadow-none"];
  if (!v) return ["shadow-md"];
  if (v.kind === "sz") {
    if (v.value === "none") return ["shadow-none"];
    if (v.value === "full") return ["shadow-2xl"];
    const s = shiftSize(v.value, v.intensity, "xs", "2xl");
    return sizeLe(s, "2xl") ? [`shadow-${s}`] : [];
  }
  if (v.kind === "col") return ["shadow-md", `shadow-${v.value}`];
  if (v.kind === "mod") return ["shadow-md", `shadow-gray-${shiftShade(v.value, v.intensity)}`];
  if (v.kind === "kw" && (v.value === "hidden" || v.value === "none")) return ["shadow-none"];
  return [];
}

function opacityValue(v: Value | null, neg: boolean): string[] {
  if (neg) return ["opacity-100"];
  if (!v) return ["opacity-50"];
  if (v.kind === "num" || v.kind === "pct") {
    const f = Number(v.value);
    const n =
      v.kind === "num" && f <= 1 && v.value.includes(".") ? Math.trunc(f * 100) : Math.trunc(f);
    return n >= 0 && n <= 100 && /^\d+(\.\d+)?$/.test(v.value) ? [`opacity-${n}`] : [];
  }
  if (v.kind === "frac") {
    const [a, b] = v.value.split("/").map(Number);
    return [`opacity-${Math.round((a! * 100) / b!)}`];
  }
  if (v.kind === "sz") return SZ_OPACITY[v.value] ? [`opacity-${SZ_OPACITY[v.value]}`] : [];
  if (v.kind === "col" && v.value === "transparent") return ["opacity-0"];
  if (v.kind === "kw" && (v.value === "hidden" || v.value === "invisible")) return ["opacity-0"];
  return [];
}

function zValue(v: Value | null, neg: boolean): string[] {
  if (neg) return ["z-0"];
  if (!v) return ["z-10"];
  if (v.kind === "num") return isInt(v.value, 0, 100) ? [`z-${v.value}`] : [];
  if (v.kind === "kw") {
    const d: Record<string, string[]> = {
      top: ["z-50"],
      first: ["z-50"],
      bottom: ["z-0"],
      last: ["z-0"],
      auto: ["z-auto"],
      negative: ["-z-10"],
    };
    return d[v.value] ?? [];
  }
  if (v.kind === "sz") {
    const d: Record<string, string[]> = {
      none: ["z-0"],
      xs: ["z-0"],
      sm: ["z-10"],
      md: ["z-20"],
      lg: ["z-30"],
      xl: ["z-40"],
      "2xl": ["z-50"],
      full: ["z-50"],
    };
    return d[v.value] ?? [];
  }
  return [];
}

function flexValue(v: Value | null, neg: boolean): string[] {
  if (neg) return ["block"];
  if (!v) return ["flex"];
  if (v.kind === "kw") {
    if (
      ["row", "col", "row-reverse", "col-reverse", "wrap", "nowrap", "wrap-reverse"].includes(
        v.value,
      )
    )
      return ["flex", `flex-${v.value}`];
    if (v.value === "center") return ["flex", "items-center", "justify-center"];
    if (["between", "around", "evenly", "start", "end"].includes(v.value))
      return ["flex", `justify-${v.value}`];
    if (v.value === "stretch" || v.value === "baseline") return ["flex", `items-${v.value}`];
    if (v.value === "grow") return ["flex-1"];
    if (v.value === "auto") return ["flex-auto"];
    if (v.value === "inline-flex" || v.value === "inline") return ["inline-flex"];
    if (v.value === "reverse") return ["flex", "flex-row-reverse"];
  }
  if (v.kind === "num") return v.value === "1" ? ["flex-1"] : [];
  if (v.kind === "sz" && v.value === "none") return ["flex-none"];
  if (v.kind === "spec") {
    const d: Record<string, string[]> = {
      center: ["flex", "items-center", "justify-center"],
      vcenter: ["flex", "items-center"],
      hcenter: ["flex", "justify-center"],
      "flex-center": ["flex", "items-center", "justify-center"],
    };
    return d[v.value] ?? [];
  }
  return [];
}

const JUSTIFY_MAP: Record<string, string> = {
  start: "start",
  end: "end",
  center: "center",
  between: "between",
  around: "around",
  evenly: "evenly",
  stretch: "stretch",
  left: "start",
  right: "end",
};
const ITEMS_MAP: Record<string, string> = {
  start: "start",
  end: "end",
  center: "center",
  baseline: "baseline",
  stretch: "stretch",
  top: "start",
  bottom: "end",
};

function simpleKw(
  prefix: string,
  allowed: Record<string, string>,
  v: Value | null,
  def: string,
): string[] {
  if (!v) return [`${prefix}-${def}`];
  if (v.kind === "kw" && allowed[v.value]) return [`${prefix}-${allowed[v.value]}`];
  if (v.kind === "spec" && ["center", "hcenter", "vcenter", "flex-center"].includes(v.value))
    return [`${prefix}-center`];
  return [];
}

/** Classes for one (property, value) pair. Mirror of emit() in semantics.py. */
export function emit(k: string, v: Value | null, neg: boolean): string[] {
  if (k.startsWith("preset:")) return split(T.presets[k.slice(7)] ?? "");
  if (v && v.kind === "lit") return [v.value];
  if (k === "gradient") {
    if (!v) return ["bg-linear-to-r"];
    if (v.kind === "col" || v.kind === "mod")
      return [`from-${v.kind === "col" ? v.value : `gray-${shiftShade(v.value, v.intensity)}`}`];
    const d = gradientDirection(v);
    return d ? [`bg-linear-to-${d}`] : [];
  }
  if (SPACING.has(k) || INSETS.has(k)) return spacingValue(k, v, neg);
  if (SIZING.has(k)) return sizingValue(k, v, neg);
  switch (k) {
    case "text":
      return textValue(v, neg);
    case "font":
      if (!v) return [];
      if (v.kind === "num" && WEIGHT_NUM[v.value]) return [`font-${WEIGHT_NUM[v.value]}`];
      if (v.kind === "mod" && v.value === "300") return ["font-light"];
      if (v.kind === "mod" && v.value === "600") return ["font-bold"];
      return textValue(v, neg);
    case "family":
      if (!v) return ["font-sans"];
      return v.kind === "kw" && ["sans", "serif", "mono"].includes(v.value)
        ? [`font-${v.value}`]
        : [];
    case "tracking": {
      if (neg) return ["tracking-normal"];
      if (!v) return ["tracking-wide"];
      if (v.kind === "kw" && ["tighter", "tight", "wide", "wider", "widest"].includes(v.value))
        return [`tracking-${v.value}`];
      if (v.kind === "sz") {
        const d: Record<string, string[]> = {
          none: ["tracking-normal"],
          xs: ["tracking-tighter"],
          sm: ["tracking-tight"],
          md: ["tracking-normal"],
          lg: ["tracking-wide"],
          xl: ["tracking-wider"],
          "2xl": ["tracking-widest"],
          full: ["tracking-widest"],
        };
        return d[v.value] ?? [];
      }
      return [];
    }
    case "leading": {
      if (neg) return ["leading-none"];
      if (!v) return ["leading-relaxed"];
      if (v.kind === "kw" && ["tight", "snug", "relaxed", "loose"].includes(v.value))
        return [`leading-${v.value}`];
      if (v.kind === "sz") {
        const d: Record<string, string[]> = {
          none: ["leading-none"],
          xs: ["leading-none"],
          sm: ["leading-tight"],
          md: ["leading-normal"],
          lg: ["leading-relaxed"],
          xl: ["leading-loose"],
          "2xl": ["leading-loose"],
        };
        return d[v.value] ?? [];
      }
      if (v.kind === "num") return isInt(v.value, 3, 10) ? [`leading-${v.value}`] : [];
      return [];
    }
    case "align":
      if (!v) return ["text-center"];
      if (
        v.kind === "kw" &&
        ["left", "center", "right", "justify", "start", "end"].includes(v.value)
      )
        return [`text-${v.value}`];
      if (v.kind === "spec" && (v.value === "center" || v.value === "hcenter"))
        return ["text-center"];
      if (v.kind === "kw" && ["top", "bottom", "baseline"].includes(v.value))
        return [`align-${v.value}`];
      if (v.kind === "spec" && v.value === "vcenter") return ["align-middle"];
      return [];
    case "transform":
      if (neg) return ["normal-case"];
      if (!v) return [];
      return v.kind === "kw" &&
        ["uppercase", "lowercase", "capitalize", "normal-case"].includes(v.value)
        ? kwText(v.value, false)
        : [];
    case "decoration":
      if (neg) return ["no-underline"];
      if (!v) return ["underline"];
      return v.kind === "kw" &&
        ["underline", "no-underline", "line-through", "overline"].includes(v.value)
        ? kwText(v.value, false)
        : [];
    case "wrap":
      if (neg) return ["text-nowrap"];
      if (!v) return ["text-wrap"];
      return v.kind === "kw" &&
        ["truncate", "nowrap", "wrap", "balance", "pretty", "break-words", "break-all"].includes(
          v.value,
        )
        ? kwText(v.value, false)
        : [];
    case "whitespace":
      if (!v || (v.kind === "kw" && v.value === "wrap")) return ["whitespace-normal"];
      return v.kind === "kw" && ["pre", "pre-wrap", "pre-line", "nowrap"].includes(v.value)
        ? [`whitespace-${v.value}`]
        : [];
    case "bg":
      if (neg || (v && v.kind === "sz" && v.value === "none")) return ["bg-transparent"];
      if (!v) return [];
      return colorClass("bg", v);
    case "border-style":
      if (!v) return ["border-solid"];
      return v.kind === "kw" && ["dashed", "dotted", "solid", "double"].includes(v.value)
        ? [`border-${v.value}`]
        : [];
    case "ring":
    case "outline":
      return ringValue(k, v, neg);
    case "shadow":
      return shadowValue(v, neg);
    case "opacity":
      return opacityValue(v, neg);
    case "z":
      return zValue(v, neg);
    case "position":
      if (!v) return ["relative"];
      return v.kind === "kw" &&
        ["absolute", "relative", "fixed", "sticky", "static"].includes(v.value)
        ? [v.value]
        : [];
    case "overflow":
    case "overflow-x":
    case "overflow-y": {
      if (neg) return [`${k}-hidden`];
      if (!v) return [`${k}-auto`];
      if (v.kind === "kw") {
        const m: Record<string, string> = {
          hidden: "hidden",
          clip: "hidden",
          scroll: "scroll",
          "overflow-auto": "auto",
          auto: "auto",
          visible: "visible",
          "overflow-visible": "visible",
        };
        return m[v.value] ? [`${k}-${m[v.value]}`] : [];
      }
      return [];
    }
    case "display":
      if (neg) return ["hidden"];
      if (!v) return ["block"];
      if (v.kind === "kw") {
        if (v.value === "visible") return ["block"];
        if (
          [
            "block",
            "inline",
            "inline-block",
            "flex",
            "inline-flex",
            "grid",
            "inline-grid",
            "hidden",
            "contents",
            "table",
          ].includes(v.value)
        )
          return [v.value];
      }
      return [];
    case "flex":
      return flexValue(v, neg);
    case "direction":
      if (!v) return ["flex-row"];
      if (v.kind === "kw" && ["row", "col", "row-reverse", "col-reverse"].includes(v.value))
        return [`flex-${v.value}`];
      if (v.kind === "kw" && v.value === "reverse") return ["flex-row-reverse"];
      return [];
    case "flexwrap":
      if (neg) return ["flex-nowrap"];
      if (!v) return ["flex-wrap"];
      return v.kind === "kw" && ["wrap", "nowrap", "wrap-reverse"].includes(v.value)
        ? [`flex-${v.value}`]
        : [];
    case "grow":
      if (
        neg ||
        (v && ((v.kind === "num" && v.value === "0") || (v.kind === "kw" && v.value === "no-grow")))
      )
        return ["grow-0"];
      return ["grow"];
    case "shrink":
      if (
        neg ||
        (v &&
          ((v.kind === "num" && v.value === "0") || (v.kind === "kw" && v.value === "no-shrink")))
      )
        return ["shrink-0"];
      return ["shrink"];
    case "justify":
      return simpleKw("justify", JUSTIFY_MAP, v, "center");
    case "items":
      return simpleKw("items", ITEMS_MAP, v, "center");
    case "self":
      return simpleKw(
        "self",
        {
          start: "start",
          end: "end",
          center: "center",
          stretch: "stretch",
          auto: "auto",
          baseline: "baseline",
        },
        v,
        "center",
      );
    case "place":
      return simpleKw(
        "place-items",
        { start: "start", end: "end", center: "center", stretch: "stretch" },
        v,
        "center",
      );
    case "grid":
      if (!v) return ["grid"];
      if (v.kind === "num") return isInt(v.value, 1, 12) ? ["grid", `grid-cols-${v.value}`] : [];
      if (v.kind === "spec" && (v.value === "center" || v.value === "flex-center"))
        return ["grid", "place-items-center"];
      if (v.kind === "kw" && v.value === "center") return ["grid", "place-items-center"];
      return [];
    case "cols":
      if (!v) return ["grid", "grid-cols-2"];
      if (v.kind === "num") return isInt(v.value, 1, 12) ? ["grid", `grid-cols-${v.value}`] : [];
      if (v.kind === "sz" && v.value === "none") return ["grid-cols-none"];
      return [];
    case "rows":
      if (!v) return ["grid", "grid-rows-2"];
      return v.kind === "num" && isInt(v.value, 1, 6) ? ["grid", `grid-rows-${v.value}`] : [];
    case "col-span":
      if (!v) return ["col-span-2"];
      if (v.kind === "num") return isInt(v.value, 1, 12) ? [`col-span-${v.value}`] : [];
      if (v.kind === "sz" && v.value === "full") return ["col-span-full"];
      return [];
    case "row-span":
      if (!v) return ["row-span-2"];
      return v.kind === "num" && isInt(v.value, 1, 6) ? [`row-span-${v.value}`] : [];
    case "order":
      if (!v) return [];
      if (v.kind === "num") return isInt(v.value, 1, 12) ? [`order-${v.value}`] : [];
      return v.kind === "kw" && (v.value === "first" || v.value === "last")
        ? [`order-${v.value}`]
        : [];
    case "transition": {
      if (neg) return ["transition-none"];
      if (!v) return ["transition"];
      if (v.kind === "kw") {
        if (["colors", "all", "opacity", "shadow", "transform"].includes(v.value))
          return [`transition-${v.value}`];
        if (v.value === "fast") return ["transition", "duration-150"];
        if (v.value === "slow") return ["transition", "duration-500"];
        if (["linear", "ease-in", "ease-out", "ease-in-out"].includes(v.value))
          return ["transition", v.value === "linear" ? "ease-linear" : v.value];
      }
      if (v.kind === "num")
        return isInt(v.value, 0, 5000) ? ["transition", `duration-${v.value}`] : [];
      if (v.kind === "unit" && v.value.endsWith("ms"))
        return ["transition", `duration-${v.value.slice(0, -2)}`];
      if (v.kind === "unit" && v.value.endsWith("s"))
        return ["transition", `duration-${Math.trunc(Number(v.value.slice(0, -1)) * 1000)}`];
      if (v.kind === "sz") {
        if (v.value === "none") return ["transition-none"];
        const d = SZ_DURATION[shiftSize(v.value, v.intensity, "xs", "2xl")];
        return d ? ["transition", `duration-${d}`] : ["transition"];
      }
      return [];
    }
    case "duration":
    case "delay": {
      if (!v) return k === "duration" ? ["duration-300"] : ["delay-150"];
      if (v.kind === "num") return isInt(v.value, 0, 5000) ? [`${k}-${v.value}`] : [];
      if (v.kind === "unit" && v.value.endsWith("ms")) return [`${k}-${v.value.slice(0, -2)}`];
      if (v.kind === "unit" && v.value.endsWith("s"))
        return [`${k}-${Math.trunc(Number(v.value.slice(0, -1)) * 1000)}`];
      if (v.kind === "kw" && v.value === "fast") return [`${k}-150`];
      if (v.kind === "kw" && v.value === "slow") return [`${k}-500`];
      if (v.kind === "sz") {
        const d = SZ_DURATION[shiftSize(v.value, v.intensity, "xs", "2xl")];
        return d ? [`${k}-${d}`] : [];
      }
      return [];
    }
    case "ease":
      if (!v) return ["ease-in-out"];
      if (v.kind === "kw" && v.value === "linear") return ["ease-linear"];
      return v.kind === "kw" && ["ease-in", "ease-out", "ease-in-out"].includes(v.value)
        ? [v.value]
        : [];
    case "animate":
      if (neg || (v && v.kind === "sz" && v.value === "none")) return ["animate-none"];
      if (!v) return ["animate-pulse"];
      return v.kind === "kw" && ["spin", "ping", "pulse", "bounce"].includes(v.value)
        ? [`animate-${v.value}`]
        : [];
    case "cursor": {
      if (!v) return ["cursor-pointer"];
      if (v.kind === "kw") {
        const m: Record<string, string> = {
          pointer: "pointer",
          "not-allowed": "not-allowed",
          wait: "wait",
          grab: "grab",
          move: "move",
          "text-cursor": "text",
          "default-cursor": "default",
          auto: "auto",
        };
        return m[v.value] ? [`cursor-${m[v.value]}`] : [];
      }
      return [];
    }
    case "select": {
      if (neg || !v) return ["select-none"];
      if (v.kind === "kw") {
        const m: Record<string, string> = {
          "select-none": "none",
          "select-all": "all",
          "select-text": "text",
          all: "all",
          "text-cursor": "text",
        };
        return m[v.value] ? [`select-${m[v.value]}`] : [];
      }
      if (v.kind === "sz" && v.value === "none") return ["select-none"];
      return [];
    }
    case "pointer":
      if (neg || !v) return ["pointer-events-none"];
      if (v.kind === "kw" && ["events-none", "events-auto", "auto"].includes(v.value))
        return [v.value !== "events-none" ? "pointer-events-auto" : "pointer-events-none"];
      if (v.kind === "sz" && v.value === "none") return ["pointer-events-none"];
      return [];
    case "object":
      if (!v) return ["object-cover"];
      return v.kind === "kw" &&
        ["cover", "contain", "fill", "center", "top", "bottom", "left", "right"].includes(v.value)
        ? [`object-${v.value}`]
        : [];
    case "aspect":
      if (!v) return ["aspect-square"];
      if (v.kind === "kw" && (v.value === "square" || v.value === "video"))
        return [`aspect-${v.value}`];
      if (v.kind === "kw" && (v.value === "aspect-auto" || v.value === "auto"))
        return ["aspect-auto"];
      if (v.kind === "frac") return [`aspect-${v.value}`];
      return [];
    case "scale": {
      if (neg) return ["scale-100"];
      if (!v) return ["scale-105"];
      if (v.kind === "num") {
        const f = Number(v.value);
        const n = f <= 3 ? Math.round(f * 100) : Math.trunc(f);
        return n >= 0 && n <= 200 ? [`scale-${n}`] : [];
      }
      if (v.kind === "pct") return isInt(v.value, 0, 200) ? [`scale-${v.value}`] : [];
      if (v.kind === "sz") {
        const d: Record<string, string[]> = {
          none: ["scale-100"],
          xs: ["scale-95"],
          sm: ["scale-105"],
          md: ["scale-110"],
          lg: ["scale-125"],
          xl: ["scale-150"],
          "2xl": ["scale-150"],
        };
        return d[v.value] ?? [];
      }
      return [];
    }
    case "rotate":
      if (neg) return ["rotate-0"];
      if (!v) return ["rotate-45"];
      if (v.kind === "num") return isInt(v.value, 0, 360) ? [`rotate-${v.value}`] : [];
      if (v.kind === "frac" && v.value === "1/2") return ["rotate-180"];
      if (v.kind === "frac" && v.value === "1/4") return ["rotate-90"];
      return [];
    case "blur":
      if (neg) return ["blur-none"];
      if (!v) return ["blur-sm"];
      if (v.kind === "sz") {
        if (v.value === "none") return ["blur-none"];
        const s = shiftSize(v.value, v.intensity, "xs", "3xl");
        return sizeLe(s, "3xl") ? [`blur-${s}`] : [];
      }
      return [];
    case "list":
      if (neg) return ["list-none"];
      if (!v) return ["list-disc"];
      if (v.kind === "kw" && (v.value === "disc" || v.value === "decimal"))
        return [`list-${v.value}`];
      if (v.kind === "sz" && v.value === "none") return ["list-none"];
      return [];
    case "visibility":
      if (neg) return ["invisible"];
      if (!v) return ["visible"];
      if (v.kind === "kw" && (v.value === "hidden" || v.value === "invisible"))
        return ["invisible"];
      if (v.kind === "kw" && v.value === "visible") return ["visible"];
      return [];
    case "columns":
      if (!v) return ["columns-2"];
      return v.kind === "num" && isInt(v.value, 1, 12) ? [`columns-${v.value}`] : [];
    case "lineclamp":
      if (neg) return ["line-clamp-none"];
      if (!v) return ["line-clamp-3"];
      return v.kind === "num" && isInt(v.value, 1, 6) ? [`line-clamp-${v.value}`] : [];
    case "container":
      return ["container", "mx-auto"];
    default:
      if (k.startsWith("border")) return borderValue(k, v, neg);
      if (k.startsWith("rounded")) return roundedValue(k, v, neg);
      return [];
  }
}

export const accepts = (k: string, v: Value): boolean => emit(k, v, false).length > 0;

const GRADIENT_DIR: Record<string, string> = {
  row: "r",
  col: "b",
  "row-reverse": "l",
  "col-reverse": "t",
  right: "r",
  left: "l",
  top: "t",
  bottom: "b",
  diagonal: "br",
  reverse: "l",
};
const GRADIENT_SPEC: Record<string, string> = {
  "top-right": "tr",
  "bottom-right": "br",
  "bottom-left": "bl",
  "top-left": "tl",
};

export function gradientDirection(v: Value): string | null {
  if (v.kind === "kw") return GRADIENT_DIR[v.value] ?? null;
  if (v.kind === "spec") return GRADIENT_SPEC[v.value] ?? null;
  return null;
}

/** Joint emission for the gradient property: direction + from/via/to colour stops. */
export function emitGradient(values: Value[]): string[] {
  const colours = values
    .filter((v) => v.kind === "col" || v.kind === "mod")
    .map((v) => (v.kind === "col" ? v.value : `gray-${shiftShade(v.value, v.intensity)}`));
  const dir = values.map(gradientDirection).find((d) => d !== null) ?? "r";
  const out = [`bg-linear-to-${dir}`];
  if (colours.length >= 1) out.push(`from-${colours[0]}`);
  if (colours.length >= 3) out.push(`via-${colours[1]}`);
  if (colours.length >= 2) out.push(`to-${colours[colours.length - 1]}`);
  return out;
}

// ---------------------------------------------------------------- variants

function match(ws: Set<string>, rules: [string, string[], string[]][]): string | null {
  for (const [variant, keys, extra] of rules) {
    if (keys.some((k) => ws.has(k)) && (extra.length === 0 || extra.some((k) => ws.has(k))))
      return variant;
  }
  return null;
}

/** Mirror of resolve_variant in lexicon.py. */
export function resolveVariant(words: string[]): string | null {
  const r = T.vars;
  const ws = new Set(words);
  if (ws.has("right") && ws.has("left"))
    return words.indexOf("right") < words.indexOf("left") ? "rtl" : "ltr";
  const strong = match(ws, r.strong);
  if (strong) return strong;
  let tier: string | null = null;
  let literal = false;
  for (const name of ["2xl", "xl", "lg", "md", "sm"]) {
    if (r.tiers[name]!.some((k) => ws.has(k))) {
      tier = name;
      break;
    }
  }
  if (tier === null && r.smLiteral.some((k) => ws.has(k))) {
    tier = "sm";
    literal = true;
  }
  if (tier === null && r.lgFallback.some((k) => ws.has(k))) tier = "lg";
  if (tier === null) return match(ws, r.weak);
  if (tier === "lg" && r.xlWords.some((w) => ws.has(w))) tier = "xl";
  if (ws.has("very") && ws.has("small")) tier = "sm";
  let upTo = false;
  for (let i = 0; i < words.length - 1; i++)
    if (words[i] === "up" && words[i + 1] === "to") upTo = true;
  const isMax = upTo || r.max.some((w) => ws.has(w));
  const isMin = (!upTo && ws.has("up")) || r.min.some((w) => ws.has(w) && w !== "up");
  if (r.only.some((w) => ws.has(w))) {
    if (tier === "sm") return "only-mobile";
    if (tier === "lg" || tier === "xl" || tier === "2xl") return "only-desktop";
  }
  if (tier === "sm" && !literal) return isMin && !isMax ? "sm" : "max-sm";
  return isMax && !isMin ? `max-${tier}` : tier;
}

/** Canonical variant stacking order: responsive, dark, group/peer, state, pseudo-element. */
const VARIANT_ORDER = [
  "sm",
  "md",
  "lg",
  "xl",
  "2xl",
  "max-sm",
  "max-md",
  "max-lg",
  "max-xl",
  "print",
  "motion-reduce",
  "landscape",
  "portrait",
  "rtl",
  "ltr",
  "dark",
  "group-hover",
  "group-focus",
  "first",
  "last",
  "odd",
  "even",
  "empty",
  "open",
  "checked",
  "required",
  "invalid",
  "disabled",
  "visited",
  "hover",
  "focus",
  "focus-visible",
  "focus-within",
  "active",
  "placeholder",
  "before",
  "after",
];
export const KNOWN_VARIANTS = new Set(VARIANT_ORDER);

export function applyVariants(classes: string[], variants: string[]): string[] {
  if (variants.length === 0) return classes;
  const only = variants.find((v) => v.startsWith("only-"));
  if (only) {
    const others = variants.filter((v) => v !== only);
    const trivial = classes.every((c) => c === "block" || c === "visible" || c === "flex");
    if (trivial)
      return applyVariants([only === "only-mobile" ? "sm:hidden" : "max-lg:hidden"], others);
    return applyVariants(classes, [...others, only === "only-mobile" ? "max-sm" : "lg"]);
  }
  const sorted = [...new Set(variants)].sort(
    (a, b) => VARIANT_ORDER.indexOf(a) - VARIANT_ORDER.indexOf(b),
  );
  const prefix = sorted.join(":");
  return classes.map((c) => `${prefix}:${c}`);
}

// ---------------------------------------------------------------- validation

const STATIC = new Set(
  "block inline-block inline flex inline-flex grid inline-grid hidden contents table flow-root static relative absolute fixed sticky isolate container mx-auto sr-only visible invisible flex-row flex-row-reverse flex-col flex-col-reverse flex-wrap flex-nowrap flex-wrap-reverse flex-1 flex-auto flex-none flex-initial grow grow-0 shrink shrink-0 italic not-italic underline overline line-through no-underline uppercase lowercase capitalize normal-case truncate text-ellipsis text-clip text-wrap text-nowrap text-balance text-pretty break-words break-all antialiased border border-solid border-dashed border-dotted border-double border-none ring outline outline-hidden outline-none transition transition-none transition-all transition-colors transition-opacity transition-shadow transition-transform ease-linear ease-in ease-out ease-in-out animate-spin animate-ping animate-pulse animate-bounce animate-none cursor-pointer cursor-default cursor-not-allowed cursor-wait cursor-text cursor-move cursor-grab cursor-auto select-none select-text select-all pointer-events-none pointer-events-auto object-cover object-contain object-fill object-center object-top object-bottom object-left object-right aspect-square aspect-video aspect-auto list-none list-disc list-decimal order-first order-last group peer grayscale blur-none whitespace-normal whitespace-nowrap whitespace-pre whitespace-pre-wrap whitespace-pre-line align-middle align-top align-bottom align-baseline grid-cols-none place-items-center place-items-start place-items-end place-items-stretch col-span-full line-clamp-none text-left text-center text-right text-justify text-start text-end scale-100 rotate-0 bg-transparent".split(
    " ",
  ),
);
const COLOR = `(?:${HUES.join("|")})-(?:${SHADES.join("|")})|white|black|transparent|current|inherit`;
const SPACE = "\\d+(?:\\.5)?|\\[\\d+(?:\\.\\d+)?(?:px|rem|em|vh|vw|ch|%)\\]";
const FRAC = "\\d+/\\d+";
const PATTERNS: [RegExp, string][] = [
  [
    /^(-?)(p|px|py|pt|pr|pb|pl|ps|pe|m|mx|my|mt|mr|mb|ml|ms|me|gap|gap-x|gap-y|space-x|space-y|top|bottom|left|right|inset)-/,
    `^-?(?:p|px|py|pt|pr|pb|pl|ps|pe|m|mx|my|mt|mr|mb|ml|ms|me|gap|gap-x|gap-y|space-x|space-y|top|bottom|left|right|inset)-(?:${SPACE}|auto|full|${FRAC})$`,
  ],
  [
    /^(w|h|size|min-w|max-w|min-h|max-h)-/,
    `^(?:w|h|size|min-w|max-w|min-h|max-h)-(?:${SPACE}|${FRAC}|auto|full|screen|min|max|fit|prose|none|dvh|svh|3xs|2xs|xs|sm|md|lg|xl|2xl|3xl|4xl|5xl|6xl|7xl)$`,
  ],
  [
    /^text-/,
    `^text-(?:xs|sm|base|lg|xl|[2-9]xl|\\[\\d+(?:\\.\\d+)?(?:px|rem|em)\\]|(?:${COLOR})(?:/\\d{1,3})?)$`,
  ],
  [/^font-/, `^font-(?:${WEIGHTS.join("|")}|sans|serif|mono)$`],
  [/^tracking-/, "^tracking-(?:tighter|tight|normal|wide|wider|widest)$"],
  [/^leading-/, "^leading-(?:none|tight|snug|normal|relaxed|loose|\\d+)$"],
  [/^bg-linear-to-/, "^bg-linear-to-(?:t|tr|r|br|b|bl|l|tl)$"],
  [/^bg-/, `^bg-(?:${COLOR})(?:/\\d{1,3})?$`],
  [
    /^border-/,
    `^border-(?:0|2|4|8|\\[\\d+px\\]|(?:t|r|b|l|x|y)(?:-(?:0|2|4|8|\\[\\d+px\\]))?|(?:t|r|b|l|x|y)-(?:${COLOR})|(?:${COLOR})(?:/\\d{1,3})?)$`,
  ],
  [
    /^rounded/,
    "^rounded(?:-(?:t|b|l|r|tl|tr|bl|br))?(?:-(?:xs|sm|md|lg|xl|2xl|3xl|4xl|full|none|\\[\\d+(?:\\.\\d+)?(?:px|rem|em)\\]))?$",
  ],
  [/^ring-/, `^ring-(?:0|1|2|4|8|(?:${COLOR})(?:/\\d{1,3})?)$`],
  [/^outline-/, `^outline-(?:0|1|2|4|8|(?:${COLOR})(?:/\\d{1,3})?)$`],
  [/^shadow-/, `^shadow-(?:2xs|xs|sm|md|lg|xl|2xl|none|(?:${COLOR})(?:/\\d{1,3})?)$`],
  [/^opacity-/, "^opacity-(?:\\d{1,2}|100)$"],
  [/^-?z-/, "^-?z-(?:\\d{1,3}|auto)$"],
  [/^overflow(-x|-y)?-/, "^overflow(?:-x|-y)?-(?:auto|hidden|visible|scroll|clip)$"],
  [/^justify-/, "^justify-(?:start|end|center|between|around|evenly|stretch)$"],
  [/^items-/, "^items-(?:start|end|center|baseline|stretch)$"],
  [/^self-/, "^self-(?:auto|start|end|center|stretch|baseline)$"],
  [/^grid-cols-/, "^grid-cols-(?:[1-9]|1[0-2])$"],
  [/^grid-rows-/, "^grid-rows-[1-6]$"],
  [/^col-span-/, "^col-span-(?:[1-9]|1[0-2])$"],
  [/^row-span-/, "^row-span-[1-6]$"],
  [/^order-/, "^order-(?:[1-9]|1[0-2])$"],
  [/^duration-/, "^duration-\\d+$"],
  [/^delay-/, "^delay-\\d+$"],
  [/^scale-/, "^scale-(?:\\d{1,3})$"],
  [/^rotate-/, "^rotate-(?:\\d{1,3})$"],
  [/^blur-/, "^blur-(?:xs|sm|md|lg|xl|2xl|3xl)$"],
  [/^aspect-/, "^aspect-\\d+/\\d+$"],
  [/^(from|via|to)-/, `^(?:from|via|to)-(?:${COLOR})(?:/\\d{1,3})?$`],
  [/^columns-/, "^columns-(?:[1-9]|1[0-2])$"],
  [/^line-clamp-/, "^line-clamp-[1-6]$"],
];
const COMPILED = PATTERNS.map(([head, re]) => [head, new RegExp(re)] as const);

/** True when `cls` (with optional variant prefixes) is in the compiled Tailwind v4 vocabulary. */
export function isValidClass(cls: string): boolean {
  const parts = cls.split(":");
  const util = parts.pop()!;
  for (const v of parts) if (!KNOWN_VARIANTS.has(v)) return false;
  if (STATIC.has(util)) return true;
  for (const [head, re] of COMPILED) if (head.test(util)) return re.test(util);
  return false;
}

/** Literal Tailwind class typed by the user ("p-4", "hover:bg-blue-500"). */
export function literalClass(text: string): string | null {
  const s = text.toLowerCase().replace(/\s+/g, "");
  return /^[a-z0-9:\-/[\].%]+$/.test(s) && (s.includes("-") || s.includes(":")) && isValidClass(s)
    ? s
    : null;
}
