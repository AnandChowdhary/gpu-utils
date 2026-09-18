import { describe, expect, it } from "vitest";
import { featurize, splitLines } from "../src/features.ts";
import { NUM_ROWS, NUM_SLOTS, SLOT_BASE, SLOTS } from "../src/keywords.ts";
import hashes from "./fixtures/features-hashes.json" with { type: "json" };

/** FNV-1a over the flat row ids; mirrors training/gpu_email/export.py:row_hash. */
function rowHash(rows: number[][]): number {
  let h = 2166136261;
  for (const row of rows) {
    for (const id of row) {
      h = Math.imul(h ^ id, 16777619) >>> 0;
    }
  }
  return h >>> 0;
}

describe("gpu-email features", () => {
  it("emits one fixed-width row per token with ids inside the table", () => {
    const f = featurize("Hi Bob,\n\nThanks!\n-- \nJohn\n+1 555 123 4567\n");
    expect(f.rows).toHaveLength(f.tokens.length);
    for (const row of f.rows) {
      expect(row).toHaveLength(NUM_SLOTS);
      for (let s = 0; s < NUM_SLOTS; s++) {
        expect(row[s]!).toBeGreaterThanOrEqual(SLOT_BASE[s]!);
        expect(row[s]!).toBeLessThan(SLOT_BASE[s]! + SLOTS[s]![1]);
        expect(row[s]!).toBeLessThan(NUM_ROWS);
      }
    }
  });

  it("splits lines with the newline token attached to its line", () => {
    const f = featurize("a\nb\r\n\nc");
    expect(splitLines(f.tokens).length).toBe(4);
    expect(f.lines.map((l) => l.text)).toEqual(["a", "b", "", "c"]);
    expect(f.lines.map((l) => l.blank)).toEqual([false, false, true, false]);
    expect(f.lines[1]!.charStart).toBe(2);
    expect(f.lines[1]!.charEnd).toBe(3);
  });

  it("detects exact rule lines", () => {
    const f = featurize("> quoted\n-- \nsig\n--\nalso delimiter\nFrom: x\n");
    expect(f.lines.map((l) => l.quotePrefixed)).toEqual([true, false, false, false, false, false]);
    expect(f.lines.map((l) => l.delimiter)).toEqual([false, true, false, true, false, false]);
    expect(f.lines[5]!.isHeader).toBe(true);
  });

  it("handles empty input", () => {
    const f = featurize("");
    expect(f.rows).toEqual([]);
    expect(f.lines).toEqual([]);
  });

  it("matches the Python featurizer on the unfamiliar set (row hashes)", () => {
    const cases = hashes as { name: string; text: string; tokens: number; hash: number }[];
    expect(cases.length).toBeGreaterThanOrEqual(60);
    for (const c of cases) {
      const f = featurize(c.text);
      expect(f.tokens.length, c.name).toBe(c.tokens);
      expect(rowHash(f.rows), c.name).toBe(c.hash);
    }
  });
});
