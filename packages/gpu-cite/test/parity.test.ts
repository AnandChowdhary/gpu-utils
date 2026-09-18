import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

/**
 * Parity fixtures are written by `pnpm export` (training/gpu_cite/export.py) into
 * model/fixtures.json: { text, rows, tags, parts, type } per case, where the logits come
 * from PyTorch running the exported int6 weights. The CPU reference path must reproduce
 * them within 1e-4; the WebGPU path is checked against the CPU path in test/gpu.test.ts
 * when an adapter is available.
 */
const TOL = 1e-4;

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}

describe("gpu-cite parity", () => {
  it("has at least 20 fixtures", () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(20);
  });

  it("loads the manifest with the expected tensors", () => {
    expect(MODEL.manifest.parameters).toBeGreaterThan(40_000);
    expect(MODEL.manifest.parameters).toBeLessThan(100_000);
    expect(MODEL.weights.length).toBe(MODEL.manifest.parameters);
    expect(MODEL.manifest.labels).toHaveLength(31);
  });

  for (const c of fixtures) {
    it(`CPU logits match PyTorch: ${JSON.stringify(c.text.slice(0, 50))}`, () => {
      const f = featurize(c.text);
      expect(f.rows).toEqual(c.rows);
      const out = forwardCpu(MODEL, f);
      expect(out.n).toBe(f.tokens.length);
      expect(maxAbsDiff(out.tags, c.tags)).toBeLessThan(TOL);
      expect(maxAbsDiff(out.parts, c.parts)).toBeLessThan(TOL);
      expect(maxAbsDiff(out.type, c.type)).toBeLessThan(TOL);
    });
  }
});
