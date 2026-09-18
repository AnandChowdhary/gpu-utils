import { describe, expect, it } from "vitest";
import { bioStartMask, bioToSpans, bioTransitions } from "../src/bio.ts";
import { viterbi } from "../src/decode.ts";
import fixture from "./fixtures/viterbi.json" with { type: "json" };

describe("viterbi cross-language fixture", () => {
  for (const [i, c] of fixture.cases.entries()) {
    it(`case ${i} matches decode.py`, () => {
      const n = c.emissions.length;
      const k = fixture.labels.length;
      const path = viterbi(
        Float32Array.from(c.emissions.flat()),
        n,
        k,
        Float32Array.from(c.transitions.flat()),
      );
      expect(Array.from(path)).toEqual(c.path);
    });
  }
});

describe("bioTransitions", () => {
  const labels = ["O", "B-A", "I-A", "B-B", "I-B"];
  it("forbids O -> I-X, B-X -> I-Y and I-X -> I-Y", () => {
    const t = bioTransitions(labels);
    const k = labels.length;
    expect(t[0 * k + 2]).toBe(-1e9);
    expect(t[1 * k + 2]).toBe(0);
    expect(t[2 * k + 2]).toBe(0);
    expect(t[3 * k + 2]).toBe(-1e9);
    expect(t[4 * k + 2]).toBe(-1e9);
    for (let from = 0; from < k; from++) {
      expect(t[from * k + 0]).toBe(0);
      expect(t[from * k + 1]).toBe(0);
    }
    expect(Array.from(bioStartMask(labels))).toEqual([0, 0, -1e9, 0, -1e9]);
  });
  it("matches the Python table in the fixture", () => {
    // fixture transitions = bio table + noise where allowed; forbidden cells are exactly -1e9
    const t = bioTransitions(fixture.labels);
    for (const c of fixture.cases) {
      for (const [i, v] of c.transitions.flat().entries()) {
        if (t[i] === -1e9) expect(v).toBe(-1e9);
        else expect(v).not.toBe(-1e9);
      }
    }
  });
});

describe("bioToSpans", () => {
  const labels = ["O", "B-A", "I-A", "B-B", "I-B"];
  it("decodes spans leniently (stray I-X starts a span)", () => {
    const spans = bioToSpans([1, 2, 0, 4, 4, 3], labels);
    expect(spans.map((s) => [s.label, s.startToken, s.endToken])).toEqual([
      ["A", 0, 2],
      ["B", 3, 5],
      ["B", 5, 6],
    ]);
  });
  it("maps to character offsets when tokens are given", () => {
    const tokens = [
      { start: 0, end: 3 },
      { start: 3, end: 4 },
      { start: 4, end: 9 },
    ];
    const spans = bioToSpans([1, 2, 3], labels, tokens);
    expect(spans.map((s) => [s.label, s.start, s.end])).toEqual([
      ["A", 0, 4],
      ["B", 4, 9],
    ]);
  });
  it("returns nothing for all-O", () => {
    expect(bioToSpans([0, 0], labels)).toEqual([]);
  });
});
