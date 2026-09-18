import { describe, expect, it } from "vitest";
import { featurize, findArxiv, findDois, findUrls, TABLE_ROWS, WIDTH } from "../src/features.ts";
import fixtures from "./fixtures/features.json" with { type: "json" };

/** Fixtures are written by `uv run python -m gpu_cite.features` from the Python featurizer. */
describe("gpu-cite featurizer parity", () => {
  for (const c of fixtures) {
    it(JSON.stringify(c.text.slice(0, 40)), () => {
      const f = featurize(c.text);
      expect(f.tokens.map((t) => [t.text, t.start, t.end])).toEqual(c.tokens);
      expect(f.rows).toEqual(c.rows);
      expect(findUrls(c.text)).toEqual(c.urls);
      expect(findDois(c.text)).toEqual(c.dois);
      expect(findArxiv(c.text)).toEqual(c.arxiv);
    });
  }

  it("emits fixed-width rows inside the table", () => {
    const f = featurize("Doe, J. (2020). Title. Journal, 1(2), 3–4.");
    for (const row of f.rows) {
      expect(row).toHaveLength(WIDTH);
      for (const id of row) expect(id).toBeGreaterThanOrEqual(0);
      for (const id of row) expect(id).toBeLessThan(TABLE_ROWS);
    }
  });

  it("handles empty input", () => {
    expect(featurize("").rows).toEqual([]);
  });
});
