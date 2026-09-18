import { type FeatureRows, hashToken, type Token, tokenize } from "@gpu-utils/runtime";

/**
 * CPU pre-pass: split a reference string into tokens and emit a fixed-width row of
 * sparse feature ids per token. Must match training/gpu_cite/features.py exactly
 * (parity enforced by test/features.test.ts and test/parity.test.ts).
 *
 * Row layout (id 0 = padding row, always zero):
 *   0 word hash · 1 consonant skeleton · 2 shape · 3 first char · 4 last char
 *   5 length bucket · 6 position bucket · 7..11 up to five flags
 */
export const WORD_BUCKETS = 1024;
export const SKEL_BUCKETS = 256;
export const SHAPES = 8;
export const CHAR_BUCKETS = 64;
export const LEN_BUCKETS = 16;
export const POS_BUCKETS = 8;
export const MAX_FLAGS = 5;
export const WIDTH = 7 + MAX_FLAGS;

const OFF_WORD = 1;
const OFF_SKEL = OFF_WORD + WORD_BUCKETS;
const OFF_SHAPE = OFF_SKEL + SKEL_BUCKETS;
const OFF_FIRST = OFF_SHAPE + SHAPES;
const OFF_LAST = OFF_FIRST + CHAR_BUCKETS;
const OFF_LEN = OFF_LAST + CHAR_BUCKETS;
const OFF_POS = OFF_LEN + LEN_BUCKETS;
const OFF_FLAG = OFF_POS + POS_BUCKETS;

const FLAG_NAMES = [
  "YEARLIKE",
  "DIGITS_SHORT",
  "DIGITS_LONG",
  "INITIAL",
  "ACRONYM",
  "MONTH",
  "CUE_IN",
  "CUE_ED",
  "CUE_VOL",
  "CUE_NO",
  "CUE_PP",
  "CUE_PROC",
  "CUE_JOURNAL",
  "CUE_PUB",
  "CUE_UNIV",
  "CUE_THESIS",
  "CUE_REPORT",
  "CUE_ACCESS",
  "CUE_DOI",
  "CUE_ARXIV",
  "CUE_AND",
  "CUE_ETAL",
  "CUE_EDITION",
  "ORDINAL",
  "CITY",
  "PARTICLE",
  "SUFFIX",
  "IN_URL",
  "IN_DOI",
  "IN_ARXIV",
  "QUOTE",
  "BRACKET",
  "DASH",
  "CUE_WEB",
  "ROMAN",
  "CUE_PART",
  "STOP",
  "NONASCII",
] as const;
type FlagName = (typeof FLAG_NAMES)[number];
const FLAG = Object.fromEntries(FLAG_NAMES.map((n, i) => [n, i])) as Record<FlagName, number>;
export const TABLE_ROWS = OFF_FLAG + FLAG_NAMES.length;

// Lexicon groups. Keep in sync with features.py (same words, same groups, same order).
const LEXICON = new Map<string, FlagName>();
function group(name: FlagName, words: string): void {
  for (const w of words.split(" ")) if (!LEXICON.has(w)) LEXICON.set(w, name);
}
group(
  "MONTH",
  "january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec",
);
group("CUE_IN", "in");
group("CUE_ED", "ed eds edited editor editors hrsg dir dirs");
group("CUE_VOL", "vol volume vols bd jahrgang tome");
group("CUE_NO", "no number issue num nr heft");
group("CUE_PP", "pp p pages page pg pgs");
group(
  "CUE_PROC",
  "proceedings proc conference conf symposium symp workshop congress meeting annual international intl ieee acm usenix",
);
group(
  "CUE_JOURNAL",
  "journal j review rev letters lett transactions trans bulletin bull annals archives acta magazine quarterly studies science nature physics physical chemistry chemical biology medicine research reports communications comm",
);
group(
  "CUE_PUB",
  "press publishers publishing publications verlag books wiley springer elsevier routledge sage oxford cambridge mit pearson mcgraw hill penguin random house academic kluwer plenum prentice addison wesley blackwell macmillan palgrave harvard princeton yale stanford reilly siam nature",
);
group(
  "CUE_UNIV",
  "university univ universität universite college institute institut school department dept laboratory lab faculty",
);
group("CUE_THESIS", "thesis dissertation phd ph msc ma master masters doctoral diss");
group("CUE_REPORT", "report technical tech rep memo memorandum working paper tr rfc");
group("CUE_ACCESS", "retrieved accessed available viewed visited cited online from");
group("CUE_DOI", "doi");
group("CUE_ARXIV", "arxiv preprint eprint biorxiv medrxiv ssrn hal abs");
group("CUE_AND", "and & und et y e");
group("CUE_ETAL", "al others");
group("CUE_EDITION", "edition edn revised");
group("ORDINAL", "st nd rd th");
group(
  "CITY",
  "new york london berlin cambridge oxford paris boston chicago heidelberg amsterdam tokyo beijing washington san francisco los angeles philadelphia dordrecht hoboken cham singapore sydney toronto delhi mumbai vienna zurich munich milan rome madrid barcelona stockholm copenhagen edinburgh dublin ny ma ca uk usa dc nj il pa",
);
group("PARTICLE", "van von de der den la le di da del du bin ibn dos das");
group("SUFFIX", "jr sr ii iii iv");
group("CUE_WEB", "web website homepage blog wikipedia github http https www html htm com org net");
group("CUE_PART", "chapter ch part sec section appendix");
group("STOP", "of the a an for on to with by");

const QUOTE_CHARS = new Set("\"'“”‘’«»‚„");
const BRACKET_CHARS = new Set("()[]{}");
const DASH_CHARS = new Set("-–—‐‑");
const ROMAN_RE = /^[ivxlc]{1,6}$/;
const YEAR_RE = /^(?:1[5-9]\d\d|20\d\d)$/;

/** Deterministic regexes shared with the decoder; ASCII classes so JS and Python agree. */
export const URL_RE = /(?:https?:\/\/|www\.)[^\s<>"'“”]+/gi;
export const DOI_RE = /10\.\d{4,9}\/[^\s"<>“”]+/gi;
export const ARXIV_RE =
  /(?:arxiv[:\s]*|abs\/|pdf\/)(\d{4}\.\d{4,5}(?:v\d+)?)|((?:astro-ph|hep-th|hep-ph|hep-ex|hep-lat|gr-qc|quant-ph|cond-mat|math-ph|nucl-th|nucl-ex|chao-dyn|alg-geom|q-alg|solv-int|math|cs|physics|nlin|q-bio|q-fin|stat|econ|eess)(?:\.[A-Za-z]{2})?\/\d{7}(?:v\d+)?)/gi;

export type Span = [number, number];

function count(s: string, ch: string): number {
  let n = 0;
  for (const c of s) if (c === ch) n++;
  return n;
}

/** Strip trailing punctuation from a URL/DOI match, keeping balanced parentheses. */
export function trimMatch(text: string, start: number, endIn: number): Span {
  let end = endIn;
  while (end > start && ".,;:".includes(text[end - 1]!)) end--;
  if (end > start && text[end - 1] === ")") {
    const s = text.slice(start, end);
    if (count(s, "(") < count(s, ")")) end--;
  }
  while (end > start && ".,;:".includes(text[end - 1]!)) end--;
  return [start, end];
}

export function findUrls(text: string): Span[] {
  const out: Span[] = [];
  for (const m of text.matchAll(URL_RE)) out.push(trimMatch(text, m.index, m.index + m[0].length));
  return out;
}

export function findDois(text: string): Span[] {
  const out: Span[] = [];
  for (const m of text.matchAll(DOI_RE)) out.push(trimMatch(text, m.index, m.index + m[0].length));
  return out;
}

export function findArxiv(text: string): Span[] {
  const out: Span[] = [];
  for (const m of text.matchAll(ARXIV_RE)) {
    const g = m[1] !== undefined ? m[1] : m[2]!;
    const start = m.index + m[0].length - g.length;
    out.push([start, start + g.length]);
  }
  return out;
}

function hasNonAscii(text: string): boolean {
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) > 127) return true;
  return false;
}

function skeleton(text: string): string {
  const low = text.toLowerCase();
  let s = "";
  for (const c of low) if (!"aeiou".includes(c)) s += c;
  return s.length > 0 ? s : `_${low}`;
}

function inSpans(spans: Span[], start: number, end: number): boolean {
  return spans.some(([s, e]) => s <= start && end <= e);
}

function tokenFlags(t: Token, inUrl: boolean, inDoi: boolean, inArxiv: boolean): number[] {
  const text = t.text;
  const low = text.toLowerCase();
  const flags: number[] = [];
  const len = Array.from(text).length;
  if (t.cls === 1) {
    if (YEAR_RE.test(text)) flags.push(FLAG.YEARLIKE);
    else if (len <= 3) flags.push(FLAG.DIGITS_SHORT);
    else if (len >= 5) flags.push(FLAG.DIGITS_LONG);
  } else if (t.cls === 0) {
    if (len === 1 && t.shape === 1) flags.push(FLAG.INITIAL);
    else if (len >= 2 && len <= 6 && t.shape === 1) flags.push(FLAG.ACRONYM);
    const g = LEXICON.get(low);
    if (g !== undefined) flags.push(FLAG[g]);
    if (ROMAN_RE.test(low) && g !== "CUE_IN") flags.push(FLAG.ROMAN);
    if (hasNonAscii(text)) flags.push(FLAG.NONASCII);
  } else if (t.cls === 4) {
    if (QUOTE_CHARS.has(text)) flags.push(FLAG.QUOTE);
    else if (BRACKET_CHARS.has(text)) flags.push(FLAG.BRACKET);
    else if (DASH_CHARS.has(text)) flags.push(FLAG.DASH);
    else if (text === "&") flags.push(FLAG.CUE_AND);
  }
  if (inUrl) flags.push(FLAG.IN_URL);
  if (inDoi) flags.push(FLAG.IN_DOI);
  if (inArxiv) flags.push(FLAG.IN_ARXIV);
  return flags.slice(0, MAX_FLAGS);
}

export function featurize(text: string): FeatureRows {
  const tokens = tokenize(text);
  const urls = findUrls(text);
  const dois = findDois(text);
  const arx = findArxiv(text);
  const n = tokens.length;
  const rows = tokens.map((t, i) => {
    const chars = Array.from(t.text);
    const row = [
      OFF_WORD + hashToken(t.text, WORD_BUCKETS),
      OFF_SKEL + hashToken(skeleton(t.text), SKEL_BUCKETS),
      OFF_SHAPE + t.shape,
      OFF_FIRST + hashToken(chars[0]!, CHAR_BUCKETS),
      OFF_LAST + hashToken(chars[chars.length - 1]!, CHAR_BUCKETS),
      OFF_LEN + Math.min(chars.length, LEN_BUCKETS - 1),
      OFF_POS + Math.floor((POS_BUCKETS * i) / n),
    ];
    const flags = tokenFlags(
      t,
      inSpans(urls, t.start, t.end),
      inSpans(dois, t.start, t.end),
      inSpans(arx, t.start, t.end),
    );
    for (const f of flags) row.push(OFF_FLAG + f);
    while (row.length < WIDTH) row.push(0);
    return row;
  });
  return { tokens, rows };
}
