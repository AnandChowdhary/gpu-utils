import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

/**
 * model/fixtures.json is written by `pnpm export` (gpu_utils_training.export.export_package)
 * in the canonical format: { cases: [{ input, rows, logits, pooled }] }, computed by the
 * Python reference on the decoded int6 weights. The featurizer must reproduce `rows`
 * byte for byte and the CPU forward pass must reproduce `logits` at 1e-4. The WGSL path
 * is checked against the same fixtures by training/tests/test_wgsl.py (lavapipe on CI).
 */
type Case = { input: string; rows: number[][]; logits: number[][]; pooled: number[] | null };
const cases = (fixtures as { cases: Case[] }).cases;

describe("__NAME__ parity", () => {
  it("ships at least 20 fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));
  it("declares the feature layout the featurizer produces", () => {
    expect(MODEL.manifest.family).toBe("scan");
    expect(MODEL.manifest.slots).toBe(3);
  });
  it.each(cases.map((c, i) => [i, c.input] as const))("case %i matches PyTorch: %s", (i) => {
    const c = cases[i]!;
    const f = featurize(c.input);
    expect(f.rows).toEqual(c.rows);
    const logits = forwardCpu(MODEL, f);
    const k = MODEL.manifest.labels.length;
    expect(logits.length).toBe(c.rows.length * k);
    let worst = 0;
    for (let t = 0; t < c.rows.length; t++) {
      for (let o = 0; o < k; o++) {
        worst = Math.max(worst, Math.abs(logits[t * k + o]! - c.logits[t]![o]!));
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });
});
