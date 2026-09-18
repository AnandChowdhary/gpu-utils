import { describe, expect, it } from "vitest";
import { FEATURE_ROWS, featurize, SLOTS } from "../src/features.ts";
import { enumEntries, fieldEntries, matchSpans, modelTokens, type Schema } from "../src/match.ts";
import { resolveTime } from "../src/time.ts";
import { read } from "./helpers.ts";

type FeatureCase = {
  text: string;
  schema: Schema;
  tokens: string[];
  rows: number[][];
};
type TimeCase = { now: string; text: string; range: [string, string] | null };
type MatchCase = {
  text: string;
  schema: Schema;
  fields: number[][];
  enums: (number | number[])[][];
};

describe("featurizer parity with Python", () => {
  const cases = read<FeatureCase[]>("test/fixtures/features.json");
  it("has the expected fixture count", () => expect(cases.length).toBeGreaterThanOrEqual(60));
  it.each(cases.map((c, i) => [i, c.text] as const))("case %i: %s", (i) => {
    const c = cases[i]!;
    const f = featurize(c.text, c.schema);
    expect(f.tokens.map((t) => t.text)).toEqual(c.tokens);
    expect(f.rows).toEqual(c.rows);
    for (const r of f.rows) {
      expect(r.length).toBeLessThanOrEqual(SLOTS);
      for (const v of r) expect(v).toBeLessThan(FEATURE_ROWS);
    }
  });
});

describe("time resolver parity with Python", () => {
  const cases = read<TimeCase[]>("test/fixtures/time.json");
  it.each(cases.map((c, i) => [i, c.text, c.now] as const))("case %i: %s @ %s", (i) => {
    const c = cases[i]!;
    expect(resolveTime(c.text, new Date(c.now))).toEqual(c.range);
  });
});

describe("matcher parity with Python", () => {
  const cases = read<MatchCase[]>("test/fixtures/match.json");
  it.each(cases.map((c, i) => [i, c.text] as const))("case %i: %s", (i) => {
    const c = cases[i]!;
    const toks = modelTokens(c.text);
    const fs = matchSpans(toks, fieldEntries(c.schema)).map((s) => [
      s.start,
      s.end,
      s.field,
      s.quality,
      s.alias ? 1 : 0,
    ]);
    const es = matchSpans(toks, enumEntries(c.schema)).map((s) => [
      s.start,
      s.end,
      s.field,
      s.value,
      s.owners,
    ]);
    expect(fs).toEqual(c.fields);
    expect(es).toEqual(c.enums);
  });
});
