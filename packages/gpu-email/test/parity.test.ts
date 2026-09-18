import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { logitWidth, MODEL } from "../src/model.ts";

/**
 * model/fixtures.json is written by `pnpm export` (gpu_utils_training.export.export_package)
 * in the canonical format: { cases: [{ input, rows, logits, pooled }] }, computed by the
 * Python reference on the decoded int6 weights. The featurizer must reproduce `rows`
 * byte for byte and the CPU forward pass (the runtime's convTaggerForward) must reproduce
 * `logits` at 1e-4. The WGSL path is checked against the same fixtures by
 * training/tests/test_wgsl.py (lavapipe on CI).
 */
type Case = { input: string; rows: number[][]; logits: number[][]; pooled: number[] | null };
const cases = (fixtures as { cases: Case[] }).cases;

describe("gpu-email parity", () => {
  it("ships at least 20 fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));

  it("manifest describes the exported conv-family model", () => {
    const m = MODEL.manifest;
    expect(m.family).toBe("conv");
    expect(m.slots).toBe(37);
    expect(m.dilations).toEqual([1, 2, 4, 8, 16, 32]);
    expect(m.labels).toHaveLength(8);
    expect(m.fields).toHaveLength(15);
    expect(m.tags).toBe(23);
    expect(m.pooled).toBe(0);
    expect(m.parameters).toBeGreaterThan(100_000);
    expect(m.parameters).toBeLessThan(250_000);
    expect(MODEL.weights.length).toBe(m.parameters);
  });

  it.each(cases.map((c, i) => [i, c.input.slice(0, 30)] as const))(
    "case %i matches PyTorch: %s",
    (i) => {
      const c = cases[i]!;
      const f = featurize(c.input);
      expect(f.rows).toEqual(c.rows);
      const logits = forwardCpu(MODEL, f);
      const w = logitWidth(MODEL);
      expect(logits.length).toBe(c.rows.length * w);
      let worst = 0;
      for (let t = 0; t < c.rows.length; t++) {
        for (let o = 0; o < w; o++) {
          worst = Math.max(worst, Math.abs(logits[t * w + o]! - c.logits[t]![o]!));
        }
      }
      expect(worst).toBeLessThan(1e-4);
    },
  );
});
