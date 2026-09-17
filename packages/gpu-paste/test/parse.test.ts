import { describe, expect, it } from "vitest";
import { featurize } from "../src/features.ts";
import { parse } from "../src/index.ts";

describe("gpu-paste", () => {
  it("emits one feature row per token", () => {
    const f = featurize("hello world");
    expect(f.rows).toHaveLength(f.tokens.length);
  });

  it("parses on the CPU reference path", async () => {
    const result = await parse("hello world", { backend: "cpu" });
    expect(result.labels).toHaveLength(3);
    expect(result.tokens.map((t) => t.text)).toEqual(["hello", " ", "world"]);
  });
});
