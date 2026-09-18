const encoder = new TextEncoder();

/** 32-bit FNV-1a over UTF-8 of the lowercased token with digits collapsed to "0", mod buckets. */
export function hashToken(text: string, buckets: number): number {
  const normalized = text.toLowerCase().replace(/\p{Nd}/gu, "0");
  let h = 0x811c9dc5;
  for (const b of encoder.encode(normalized)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h % buckets;
}
