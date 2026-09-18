import { describe, expect, it } from "vitest";
import { featurize } from "../src/features.ts";
import { parse } from "../src/index.ts";

describe("__NAME__", () => {
  it("emits one feature row per token", () => {
    const f = featurize("hello world");
    expect(f.rows).toHaveLength(f.tokens.length);
  });

  it("parses on the CPU reference path", async () => {
    const result = await parse("hello world", { backend: "cpu" });
    expect(result.labels).toHaveLength(3);
    expect(result.tokens.map((t) => t.text)).toEqual(["hello", " ", "world"]);
    for (const s of result.spans) expect("hello world".slice(s.start, s.end)).toBe(s.text);
  });

  it("handles empty input", async () => {
    const result = await parse("", { backend: "cpu" });
    expect(result.labels).toEqual([]);
    expect(result.spans).toEqual([]);
  });
});
