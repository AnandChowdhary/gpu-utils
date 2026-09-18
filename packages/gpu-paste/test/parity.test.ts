import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

/**
 * model/fixtures.json is written by `pnpm export` (gpu_utils_training.export.export_package)
 * in the canonical format `{ cases: [{ input, rows, logits, pooled }] }`, computed by the
 * Python reference on the decoded int6 weights. For gpu-paste `logits` are the BIO span
 * head ([tokens, labels]) and `pooled` the kind head ([kinds]). The featurizer must
 * reproduce `rows` byte for byte and the CPU forward pass both outputs at 1e-4. The WGSL
 * path is checked against the same fixtures by training/tests/test_wgsl.py (lavapipe).
 */
type Case = {
  input: string;
  rows: number[][];
  logits: number[][];
  pooled: number[] | null;
  tokens?: string[];
};
const cases = (fixtures as { cases: Case[] }).cases;
const TOL = 1e-4;

describe("gpu-paste parity", () => {
  it("ships at least 20 fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));

  it("declares the scan family with the featurizer's layout and both heads", () => {
    expect(MODEL.manifest.family).toBe("scan");
    expect(MODEL.manifest.slots).toBe(10);
    expect(MODEL.manifest.tags).toBe(MODEL.manifest.labels.length);
    expect(MODEL.manifest.pooled).toBe(MODEL.manifest.kinds.length);
  });

  it.each(cases.map((c, i) => [i, c.input.slice(0, 40)] as const))(
    "case %i matches PyTorch: %s",
    (i) => {
      const c = cases[i]!;
      const f = featurize(c.input);
      if (c.tokens) expect(f.tokens.map((t) => t.text)).toEqual(c.tokens);
      expect(f.rows).toEqual(c.rows);
      if (c.rows.length === 0) return;
      const out = forwardCpu(MODEL, f.rows);
      const k = MODEL.manifest.labels.length;
      expect(out.span.length).toBe(c.rows.length * k);
      let worst = 0;
      for (let t = 0; t < c.rows.length; t++) {
        for (let o = 0; o < k; o++) {
          worst = Math.max(worst, Math.abs(out.span[t * k + o]! - c.logits[t]![o]!));
        }
      }
      expect(c.pooled).not.toBeNull();
      expect(out.kind.length).toBe(c.pooled!.length);
      for (let o = 0; o < c.pooled!.length; o++) {
        worst = Math.max(worst, Math.abs(out.kind[o]! - c.pooled![o]!));
      }
      expect(worst).toBeLessThan(TOL);
    },
  );
});
