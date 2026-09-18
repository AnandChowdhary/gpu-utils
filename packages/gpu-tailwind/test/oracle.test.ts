import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { decodeLabels } from "../src/decode.ts";
import { featurize } from "../src/features.ts";

/**
 * Generator/compiler parity: compiling the generator's gold tags must reproduce the
 * generator's gold classes. training/data/oracle.jsonl is a committed 400-example
 * sample (seed 3) written by `uv run python -m gpu_tailwind.data`.
 */
const rows = readFileSync(resolve(import.meta.dirname, "../training/data/oracle.jsonl"), "utf8")
  .trim()
  .split("\n")
  .map(
    (l) =>
      JSON.parse(l) as { text: string; labels: string[]; boundary: number[]; classes: string[] },
  );

describe("oracle", () => {
  it("compiler reproduces gold classes from gold tags", () => {
    const failures: string[] = [];
    for (const r of rows) {
      const f = featurize(r.text);
      const got = decodeLabels(f, r.labels, r.boundary.map(Boolean)).classes;
      if ([...got].sort().join(" ") !== [...r.classes].sort().join(" ")) {
        failures.push(`${r.text}\n   want ${r.classes.join(" ")}\n   got  ${got.join(" ")}`);
      }
    }
    expect(failures.slice(0, 15).join("\n"), `${failures.length}/${rows.length} mismatches`).toBe(
      "",
    );
  });
});
