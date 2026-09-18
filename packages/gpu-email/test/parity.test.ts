import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { logitWidth, MODEL } from "../src/model.ts";

/**
 * Parity fixtures are written by `pnpm export` (training/gpu_email/export.py) into
 * model/fixtures.json: { text, features, logits } per case, computed by PyTorch with
 * the int6-quantized weights. The CPU reference path must reproduce the feature rows
 * exactly and the logits within 1e-4. The WebGPU path is checked against the CPU path
 * in the browser (see README).
 */
interface Fixture {
  text: string;
  features: number[][];
  logits: number[][];
}

const cases = fixtures as Fixture[];

describe("gpu-email parity", () => {
  it("has at least 20 fixture cases", () => {
    expect(cases.length).toBeGreaterThanOrEqual(20);
  });

  it("manifest describes the exported model", () => {
    expect(MODEL.manifest.labels).toHaveLength(8);
    expect(MODEL.manifest.fields).toHaveLength(15);
    expect(MODEL.manifest.parameters).toBeGreaterThan(100_000);
    expect(MODEL.manifest.parameters).toBeLessThan(250_000);
    expect(MODEL.weights.length).toBe(MODEL.manifest.parameters);
  });

  for (const [i, c] of cases.entries()) {
    it(`case ${i}: feature rows match Python (${JSON.stringify(c.text.slice(0, 30))})`, () => {
      const f = featurize(c.text);
      expect(f.rows).toEqual(c.features);
    });

    it(`case ${i}: CPU logits match PyTorch at 1e-4`, () => {
      const f = featurize(c.text);
      const logits = forwardCpu(MODEL, f);
      const w = logitWidth(MODEL);
      expect(logits.length).toBe(c.logits.length * w);
      let maxDiff = 0;
      for (let t = 0; t < c.logits.length; t++) {
        for (let j = 0; j < w; j++) {
          const d = Math.abs(logits[t * w + j]! - c.logits[t]![j]!);
          if (d > maxDiff) maxDiff = d;
        }
      }
      expect(maxDiff).toBeLessThan(1e-4);
    });
  }
});
