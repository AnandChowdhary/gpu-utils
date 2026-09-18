import { type FeatureRows, hashToken, tokenize } from "@gpu-utils/runtime";

/**
 * CPU pre-pass: split text into tokens and emit sparse feature ids per token.
 * Must match training/__SNAKE__/features.py exactly. Parity is enforced by
 * test/parity.test.ts against fixtures exported from Python. Each feature family
 * owns a block of ids in one embedding table so they never collide.
 */
const HASH_BUCKETS = 1024;
const OFFSET = { word: 0, shape: HASH_BUCKETS, length: HASH_BUCKETS + 8 } as const;
export const FEATURE_ROWS = HASH_BUCKETS + 8 + 16;
export const SLOTS = 3;

export function featurize(text: string): FeatureRows {
  const tokens = tokenize(text);
  return {
    tokens,
    rows: tokens.map((t) => [
      OFFSET.word + hashToken(t.text, HASH_BUCKETS),
      OFFSET.shape + t.shape,
      OFFSET.length + Math.min(t.text.length, 15),
    ]),
  };
}
