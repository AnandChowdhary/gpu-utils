import {
  CharClass,
  type FeatureRows,
  hashToken,
  Shape,
  type Token,
  tokenize,
} from "@gpu-utils/runtime";
import {
  EDGES_DIGITS,
  EDGES_HEADER_RUN,
  EDGES_LINE_IDX,
  EDGES_LINE_LEN,
  EDGES_PARA,
  EDGES_PARA_POS,
  EDGES_SINCE_ATTRIB,
  EDGES_UNTIL_QUOTEISH,
  EDGES_WORD_COUNT,
  FIRST_WORD_BUCKETS,
  KW_CLOSING_PHRASES,
  KW_CLOSING_WORDS,
  KW_CONTACT_WORDS,
  KW_DATE_WORDS,
  KW_DISCLAIMER,
  KW_FORWARD,
  KW_GREETING_PHRASES,
  KW_GREETING_WORDS,
  KW_HEADER,
  KW_MOBILE,
  KW_ORIGINAL,
  KW_WROTE,
  LAST_WORD_BUCKETS,
  NUM_SLOTS,
  SLOT_BASE,
  SUFFIX_BUCKETS,
  WORD_BUCKETS,
} from "./keywords.ts";

/**
 * CPU pre-pass: split text into tokens and emit a fixed-width row of NUM_SLOTS sparse
 * feature ids per token. Mirrors training/gpu_email/features.py line by line; the
 * keyword tables and slot layout are generated from the Python module into keywords.ts.
 * Parity is enforced by test/features.test.ts and test/parity.test.ts.
 */

export const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/;
export const URL_RE = /(?:https?:\/\/[^\s<>"')\]]+|www\.[A-Za-z0-9-]+\.[^\s<>"')\]]+)/;

const KW_CONTACT_SET = new Set<string>(KW_CONTACT_WORDS);
const KW_CLOSING_SET = new Set<string>(KW_CLOSING_WORDS);
const KW_GREETING_SET = new Set<string>(KW_GREETING_WORDS);
const KW_DATE_SET = new Set<string>(KW_DATE_WORDS);

const LETTER_RE = /^\p{L}$/u;
const DIGIT_RE = /^\p{Nd}$/u;

export interface LineInfo {
  /** Token index range [start, end); the newline token belongs to its line. */
  start: number;
  end: number;
  /** Character range of the line text without its newline. */
  charStart: number;
  charEnd: number;
  text: string;
  lower: string;
  blank: boolean;
  quotePrefixed: boolean;
  delimiter: boolean;
  isHeader: boolean;
  isAttribMarker: boolean;
  isForwardMarker: boolean;
}

export interface EmailFeatures extends FeatureRows {
  lines: LineInfo[];
}

function bucket(v: number, edges: readonly number[]): number {
  for (let i = 0; i < edges.length; i++) if (v <= edges[i]!) return i;
  return edges.length;
}

function containsAny(hay: string, needles: readonly string[]): boolean {
  for (const n of needles) if (hay.includes(n)) return true;
  return false;
}

function startsWithAny(hay: string, needles: readonly string[]): boolean {
  for (const n of needles) if (hay.startsWith(n)) return true;
  return false;
}

function stripSpaces(s: string): string {
  let i = 0;
  let j = s.length;
  while (i < j && (s[i] === " " || s[i] === "\t")) i++;
  while (j > i && (s[j - 1] === " " || s[j - 1] === "\t")) j--;
  return s.slice(i, j);
}

function stripPrefix(lower: string): string {
  let i = 0;
  while (i < lower.length && " \t>*|".includes(lower[i]!)) i++;
  return lower.slice(i);
}

/** [start, end) token ranges, one per line. */
export function splitLines(tokens: Token[]): [number, number][] {
  const lines: [number, number][] = [];
  let start = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i]!.cls === CharClass.Newline) {
      lines.push([start, i + 1]);
      start = i + 1;
    }
  }
  if (start < tokens.length) lines.push([start, tokens.length]);
  return lines;
}

function lineInfo(tokens: Token[], start: number, end: number): LineInfo {
  let text = "";
  let blank = true;
  for (let i = start; i < end; i++) {
    const t = tokens[i]!;
    if (t.cls === CharClass.Newline) continue;
    text += t.text;
    if (t.cls !== CharClass.Space) blank = false;
  }
  const lower = text.toLowerCase();
  const stripped = stripSpaces(lower);
  const body = stripPrefix(lower);
  const charStart = tokens[start]!.start;
  return {
    start,
    end,
    charStart,
    charEnd: charStart + text.length,
    text,
    lower,
    blank,
    quotePrefixed: stripped.startsWith(">"),
    delimiter: stripped === "--",
    isHeader: startsWithAny(body, KW_HEADER),
    isAttribMarker: containsAny(lower, KW_WROTE) || containsAny(lower, KW_ORIGINAL),
    isForwardMarker: containsAny(lower, KW_FORWARD),
  };
}

function lastCharClass(text: string): number {
  let j = text.length;
  while (j > 0 && (text[j - 1] === " " || text[j - 1] === "\t")) j--;
  if (j === 0) return 0;
  // last code point (handles surrogate pairs like Python's text[-1])
  const chars = Array.from(text.slice(Math.max(0, j - 2), j));
  const c = chars[chars.length - 1]!;
  if (c === ":" || c === "：") return 1;
  if (c === ",") return 2;
  if (c === ".") return 3;
  if (c === "?" || c === "!") return 4;
  if (c === ";") return 5;
  if (LETTER_RE.test(c)) return 6;
  if (DIGIT_RE.test(c)) return 7;
  return 8;
}

function suffix3(text: string): string {
  const cps = Array.from(text);
  return cps.length <= 3 ? text : cps.slice(-3).join("");
}

export function featurize(text: string): EmailFeatures {
  const tokens = tokenize(text);
  return featurizeTokens(tokens);
}

export function featurizeTokens(tokens: Token[]): EmailFeatures {
  const n = tokens.length;
  const rows: number[][] = [];
  const ranges = splitLines(tokens);
  const lines = ranges.map(([s, e]) => lineInfo(tokens, s, e));
  if (n === 0) return { tokens, rows, lines };
  const numLines = lines.length;

  const nonblankBelow = new Int32Array(numLines);
  let acc = 0;
  for (let li = numLines - 1; li >= 0; li--) {
    nonblankBelow[li] = acc;
    if (!lines[li]!.blank) acc++;
  }
  const paraIndex = new Int32Array(numLines);
  const posInPara = new Int32Array(numLines);
  let paraCount = 0;
  let inPara = false;
  for (let li = 0; li < numLines; li++) {
    if (lines[li]!.blank) {
      inPara = false;
      paraIndex[li] = paraCount;
      posInPara[li] = 0;
    } else {
      if (!inPara) {
        paraCount++;
        inPara = true;
        posInPara[li] = 0;
      } else {
        posInPara[li] = posInPara[li - 1]! + 1;
      }
      paraIndex[li] = paraCount - 1;
    }
  }
  const untilBlank = new Int32Array(numLines);
  let run = 0;
  for (let li = numLines - 1; li >= 0; li--) {
    if (lines[li]!.blank) {
      run = 0;
      untilBlank[li] = 0;
    } else {
      untilBlank[li] = run;
      run++;
    }
  }
  const untilQuoteish = new Int32Array(numLines);
  let nxt = -1;
  for (let li = numLines - 1; li >= 0; li--) {
    const line = lines[li]!;
    if (nxt >= 0) untilQuoteish[li] = bucket(nxt - li, EDGES_UNTIL_QUOTEISH) + 1;
    if (line.quotePrefixed || line.isAttribMarker || line.isForwardMarker || line.isHeader)
      nxt = li;
  }

  let quoteAbove = false;
  let delimAbove = false;
  let attribAbove = false;
  let headerAbove = false;
  let lastAttrib = -1;
  let headerRun = 0;
  for (let li = 0; li < numLines; li++) {
    const line = lines[li]!;
    const flags = (line.quotePrefixed ? 1 : 0) | (line.delimiter ? 2 : 0) | (line.blank ? 4 : 0);
    const fromTop = bucket(li, EDGES_LINE_IDX);
    const fromBottom = bucket(nonblankBelow[li]!, EDGES_LINE_IDX);
    const lineLen = bucket(line.text.length, EDGES_LINE_LEN);

    let firstWord = 0;
    let firstShape = 8;
    let lastWord = 0;
    let wordCount = 0;
    let letterTokens = 0;
    let titleTokens = 0;
    let digitChars = 0;
    let letterChars = 0;
    let kwContact = 0;
    let kwClosing = 0;
    let kwGreeting = 0;
    let kwDate = 0;
    let firstNonspaceSeen = false;
    for (let i = line.start; i < line.end; i++) {
      const t = tokens[i]!;
      if (t.cls === CharClass.Newline) continue;
      if (t.cls !== CharClass.Space && !firstNonspaceSeen) {
        firstNonspaceSeen = true;
        firstShape = t.shape;
      }
      if (t.cls === CharClass.Letter || t.cls === CharClass.Digit) {
        wordCount++;
        const h = hashToken(t.text, FIRST_WORD_BUCKETS - 1) + 1;
        if (firstWord === 0) firstWord = h;
        lastWord = hashToken(t.text, LAST_WORD_BUCKETS - 1) + 1;
      }
      if (t.cls === CharClass.Letter) {
        letterTokens++;
        letterChars += Array.from(t.text).length;
        if (t.shape === Shape.Title || t.shape === Shape.Upper) titleTokens++;
        const w = t.text.toLowerCase();
        if (KW_CONTACT_SET.has(w)) kwContact = 1;
        if (KW_CLOSING_SET.has(w)) kwClosing = 1;
        if (KW_GREETING_SET.has(w)) kwGreeting = 1;
        if (KW_DATE_SET.has(w)) kwDate = 1;
      } else if (t.cls === CharClass.Digit) {
        digitChars += Array.from(t.text).length;
      }
    }
    const lower = line.lower;
    if (containsAny(lower, KW_CLOSING_PHRASES)) kwClosing = 1;
    if (containsAny(lower, KW_GREETING_PHRASES)) kwGreeting = 1;
    const hasEmail = EMAIL_RE.test(line.text) ? 1 : 0;
    const hasUrl = URL_RE.test(line.text) ? 2 : 0;
    const phoneLike = digitChars >= 7 && digitChars <= 20 && letterChars <= 12 ? 4 : 0;
    const contentFlags = hasEmail | hasUrl | phoneLike;
    const wc = bucket(wordCount, EDGES_WORD_COUNT);
    let titleFrac: number;
    if (letterTokens === 0) titleFrac = 0;
    else if (titleTokens === 0) titleFrac = 1;
    else if (titleTokens * 3 < letterTokens) titleFrac = 2;
    else if (titleTokens * 3 < letterTokens * 2) titleFrac = 3;
    else if (titleTokens < letterTokens) titleFrac = 4;
    else titleFrac = 5;
    const digitsB = bucket(digitChars, EDGES_DIGITS);
    const prevBlank = li > 0 && lines[li - 1]!.blank;
    const nextBlank = li + 1 < numLines && lines[li + 1]!.blank;
    const neighbours =
      (prevBlank ? 1 : 0) | (nextBlank ? 2 : 0) | (li > 0 ? 4 : 0) | (li + 1 < numLines ? 8 : 0);
    const kwWrote = containsAny(lower, KW_WROTE) ? 1 : 0;
    const kwOriginal = containsAny(lower, KW_ORIGINAL) ? 1 : 0;
    const kwForward = containsAny(lower, KW_FORWARD) ? 1 : 0;
    const kwMobile = containsAny(lower, KW_MOBILE) ? 1 : 0;
    const kwHeader = line.isHeader ? 1 : 0;
    const kwDisclaimer = containsAny(lower, KW_DISCLAIMER) ? 1 : 0;

    const ctxAbove =
      (quoteAbove ? 1 : 0) | (delimAbove ? 2 : 0) | (attribAbove ? 4 : 0) | (headerAbove ? 8 : 0);
    if (line.isAttribMarker || line.isForwardMarker) lastAttrib = li;
    const sinceAttrib = lastAttrib < 0 ? 0 : bucket(li - lastAttrib, EDGES_SINCE_ATTRIB) + 1;
    const sinceBlank = bucket(posInPara[li]!, EDGES_PARA_POS);
    const untilB = bucket(untilBlank[li]!, EDGES_PARA_POS);
    const headerRunB = bucket(headerRun, EDGES_HEADER_RUN);
    const paraTop = bucket(paraIndex[li]!, EDGES_PARA);
    const paraBottom = bucket(Math.max(0, paraCount - 1 - paraIndex[li]!), EDGES_PARA);

    const lineVals = [
      flags,
      fromTop,
      fromBottom,
      lineLen,
      firstWord,
      firstShape,
      lastWord,
      lastCharClass(line.text),
      contentFlags,
      wc,
      titleFrac,
      digitsB,
      neighbours,
      kwWrote,
      kwOriginal,
      kwForward,
      kwMobile,
      kwHeader,
      kwDisclaimer,
      kwContact,
      kwClosing,
      kwGreeting,
      kwDate,
      ctxAbove,
      sinceAttrib,
      sinceBlank,
      untilB,
      untilQuoteish[li]!,
      headerRunB,
      paraTop,
      paraBottom,
    ];
    const count = line.end - line.start;
    for (let k = 0, i = line.start; i < line.end; k++, i++) {
      const t = tokens[i]!;
      const row = new Array<number>(NUM_SLOTS);
      row[0] = hashToken(t.text, WORD_BUCKETS);
      row[1] = t.shape;
      row[2] = Math.min(t.text.length, 15);
      row[3] = t.cls;
      row[4] = hashToken(suffix3(t.text), SUFFIX_BUCKETS);
      let pos: number;
      if (t.cls === CharClass.Newline) pos = 8;
      else if (
        k === count - 1 ||
        (k === count - 2 && tokens[line.end - 1]!.cls === CharClass.Newline)
      )
        pos = 7;
      else pos = bucket(k, [0, 1, 2, 3, 4, 7]);
      row[5] = pos;
      for (let s = 0; s < lineVals.length; s++) row[6 + s] = lineVals[s]!;
      for (let s = 0; s < NUM_SLOTS; s++) row[s] = row[s]! + SLOT_BASE[s]!;
      rows.push(row);
    }
    if (line.quotePrefixed) quoteAbove = true;
    if (line.delimiter) delimAbove = true;
    if (line.isAttribMarker) attribAbove = true;
    if (line.isHeader) {
      headerAbove = true;
      headerRun++;
    } else if (!line.blank) {
      headerRun = 0;
    }
  }
  return { tokens, rows, lines };
}
