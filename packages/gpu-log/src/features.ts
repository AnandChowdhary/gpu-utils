import { type FeatureRows, hashToken, type Token, tokenize } from "@gpu-utils/runtime";

/**
 * CPU pre-pass for one log line: character-class tokens plus FEATURE_COUNT sparse ids per
 * token, each pre-offset into the single flat embedding table. Must match
 * training/gpu_log/features.py byte for byte (checked by test/parity.test.ts).
 */
const WORD_BUCKETS = 1024;
const PREFIX_BUCKETS = 512;

/** (name, rows) in table order; the offsets below are their prefix sums. */
export const FEATURE_SIZES: readonly [string, number][] = [
  ["word", WORD_BUCKETS],
  ["prefix", PREFIX_BUCKETS],
  ["shape", 8],
  ["first", 129],
  ["last", 129],
  ["len", 16],
  ["pos", 20],
  ["rpos", 12],
  ["col", 10],
];
export const FEATURE_COUNT = FEATURE_SIZES.length;
const OFFSETS: number[] = [];
let acc = 0;
for (const [, size] of FEATURE_SIZES) {
  OFFSETS.push(acc);
  acc += size;
}
export const EMBED_ROWS = acc;

const OFF_WORD = OFFSETS[0]!;
const OFF_PREFIX = OFFSETS[1]!;
const OFF_SHAPE = OFFSETS[2]!;
const OFF_FIRST = OFFSETS[3]!;
const OFF_LAST = OFFSETS[4]!;
const OFF_LEN = OFFSETS[5]!;
const OFF_POS = OFFSETS[6]!;
const OFF_RPOS = OFFSETS[7]!;
const OFF_COL = OFFSETS[8]!;

const DIGIT = /^\p{Nd}/u;

/** First code point: ASCII code, non-ASCII -> 128, decimal digits collapsed to "0" (like hashToken). */
function charBucket(ch: string): number {
  if (DIGIT.test(ch)) return 48;
  const code = ch.codePointAt(0)!;
  return code < 128 ? code : 128;
}

function posBucket(i: number): number {
  if (i < 16) return i;
  if (i < 32) return 16;
  if (i < 64) return 17;
  if (i < 128) return 18;
  return 19;
}

function rposBucket(r: number): number {
  if (r < 8) return r;
  if (r < 16) return 8;
  if (r < 32) return 9;
  if (r < 64) return 10;
  return 11;
}

function colBucket(c: number): number {
  if (c === 0) return 0;
  if (c <= 8) return 1;
  if (c <= 16) return 2;
  if (c <= 24) return 3;
  if (c <= 32) return 4;
  if (c <= 48) return 5;
  if (c <= 64) return 6;
  if (c <= 96) return 7;
  if (c <= 160) return 8;
  return 9;
}

/** Python `text[:3]` on code points. */
function prefix3(text: string): string {
  let out = "";
  let count = 0;
  for (const ch of text) {
    if (count === 3) break;
    out += ch;
    count++;
  }
  return out;
}

/** Python `text[-1]` on code points. */
function lastChar(text: string): string {
  const cps = Array.from(text);
  return cps[cps.length - 1]!;
}

/** Writes the FEATURE_COUNT ids of token `i` into `out` starting at `at`. */
export function writeTokenFeatures(tokens: Token[], i: number, out: Uint32Array, at: number): void {
  const t = tokens[i]!;
  const text = t.text;
  const n = tokens.length;
  out[at] = OFF_WORD + hashToken(text, WORD_BUCKETS);
  out[at + 1] = OFF_PREFIX + hashToken(text.length > 3 ? prefix3(text) : text, PREFIX_BUCKETS);
  out[at + 2] = OFF_SHAPE + t.shape;
  out[at + 3] = OFF_FIRST + charBucket(text);
  out[at + 4] = OFF_LAST + charBucket(text.length === 1 ? text : lastChar(text));
  out[at + 5] = OFF_LEN + Math.min(Array.from(text).length, 15);
  out[at + 6] = OFF_POS + posBucket(i);
  out[at + 7] = OFF_RPOS + rposBucket(n - 1 - i);
  out[at + 8] = OFF_COL + colBucket(t.start);
}

/** Feature rows for already-tokenized text. Mirrors `featurize_tokens()` in Python. */
export function featurizeTokens(tokens: Token[]): number[][] {
  const rows: number[][] = [];
  const buf = new Uint32Array(FEATURE_COUNT);
  for (let i = 0; i < tokens.length; i++) {
    writeTokenFeatures(tokens, i, buf, 0);
    rows.push(Array.from(buf));
  }
  return rows;
}

/** Featurizes a single line (no newlines). Mirrors `featurize()` in Python. */
export function featurize(line: string): FeatureRows {
  const tokens = tokenize(line);
  return { tokens, rows: featurizeTokens(tokens) };
}
