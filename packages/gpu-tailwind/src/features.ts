import { type FeatureRows, hashToken, tokenize } from "@gpu-utils/runtime";

/**
 * CPU pre-pass: split text into tokens and emit 7 sparse feature ids per token.
 * Must match training/gpu_tailwind/features.py exactly (parity fixtures in
 * model/fixtures.json). Ids index one shared embedding table of ROWS rows:
 * word hash | consonant skeleton hash | prefix hash | suffix hash | shape | length | class.
 */
export const WORD = 1024;
export const SKEL = 256;
export const PRE = 128;
export const SUF = 128;
export const OFF_SKEL = WORD;
export const OFF_PRE = OFF_SKEL + SKEL;
export const OFF_SUF = OFF_PRE + PRE;
export const OFF_SHAPE = OFF_SUF + SUF;
export const OFF_LEN = OFF_SHAPE + 8;
export const OFF_CLS = OFF_LEN + 16;
export const ROWS = OFF_CLS + 5;
export const WIDTH = 7;

/** Lowercase, drop vowels after the first character, collapse repeated characters. */
export function skeleton(text: string): string {
  const t = text.toLowerCase();
  if (t.length === 0) return "";
  const s = t[0] + t.slice(1).replace(/[aeiou]/g, "");
  return s.replace(/(.)\1+/g, "$1");
}

export function featurize(text: string): FeatureRows {
  const tokens = tokenize(text);
  const rows = tokens.map((t) => {
    // Python slices by code point; Array.from keeps the same semantics for non-BMP text.
    const cps = Array.from(t.text);
    const len = cps.length;
    return [
      hashToken(t.text, WORD),
      OFF_SKEL + hashToken(skeleton(t.text), SKEL),
      OFF_PRE + hashToken(cps.slice(0, 3).join(""), PRE),
      OFF_SUF + hashToken(cps.slice(Math.max(0, len - 3)).join(""), SUF),
      OFF_SHAPE + t.shape,
      OFF_LEN + Math.min(len, 15),
      OFF_CLS + t.cls,
    ];
  });
  return { tokens, rows };
}
