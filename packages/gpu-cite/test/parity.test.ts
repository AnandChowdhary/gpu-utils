import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize, WIDTH } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

/**
 * model/fixtures.json is written by `pnpm export` (gpu_utils_training.export.export_package)
 * in the canonical format { cases: [{ input, rows, logits, pooled }] }: per token the 31 BIO
 * emissions followed by the 3 name-part logits, plus the 8 pooled type logits, computed by
 * the Python family forward on the decoded int6 weights. The featurizer must reproduce
 * `rows` byte for byte and the CPU forward pass must reproduce the logits at 1e-4; the WGSL
 * path is checked against the same fixtures by training/tests/test_wgsl.py (lavapipe on CI).
 */
type Case = { input: string; rows: number[][]; logits: number[][]; pooled: number[] | null };
const cases = (fixtures as { cases: Case[] }).cases;
const TOL = 1e-4;

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}

describe("gpu-cite parity", () => {
  it("ships at least 20 fixtures", () => {
    expect(cases.length).toBeGreaterThanOrEqual(20);
  });

  it("loads a scan-family manifest with the decoder's extra tensor", () => {
    const m = MODEL.manifest;
    expect(m.family).toBe("scan");
    expect(m.slots).toBe(WIDTH);
    expect(m.paddingId).toBe(0);
    expect(m.labels).toHaveLength(31);
    expect(m.tags).toBe(m.labels.length + m.nameparts.length);
    expect(m.pooled).toBe(m.types.length);
    expect(m.tensors.some((t) => t.name === "trans")).toBe(true);
    expect(MODEL.weights.length).toBe(m.parameters);
    expect(m.parameters).toBeGreaterThan(40_000);
    expect(m.parameters).toBeLessThan(100_000);
  });

  for (const c of cases) {
    it(`CPU logits match PyTorch: ${JSON.stringify(c.input.slice(0, 50))}`, () => {
      const K = MODEL.manifest.labels.length;
      const P = MODEL.manifest.nameparts.length;
      const f = featurize(c.input);
      expect(f.rows).toEqual(c.rows);
      const out = forwardCpu(MODEL, f);
      expect(out.n).toBe(c.rows.length);
      expect(
        maxAbsDiff(
          out.tags,
          c.logits.flatMap((row) => row.slice(0, K)),
        ),
      ).toBeLessThan(TOL);
      expect(
        maxAbsDiff(
          out.parts,
          c.logits.flatMap((row) => row.slice(K, K + P)),
        ),
      ).toBeLessThan(TOL);
      expect(maxAbsDiff(out.type, c.pooled ?? [])).toBeLessThan(TOL);
    });
  }
});
