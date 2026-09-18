import { describe, expect, it } from "vitest";
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import type { Schema } from "../src/match.ts";
import { MODEL } from "../src/model.ts";
import { read } from "./helpers.ts";

type Fixture = {
  text: string;
  schema: Schema;
  tokens: string[];
  rows: number[][];
  logits: number[][];
};

/**
 * model/fixtures.json is written by training/gpu_view/export.py: inputs, feature rows and
 * logits from a float64 reference over the dequantized int6 weights. The CPU path must
 * reproduce them at 1e-4. The WGSL path is checked against the CPU path in
 * test/wgsl-parity (needs a WebGPU adapter) and by the Python wgpu harness.
 */
describe("gpu-view parity", () => {
  const cases = read<Fixture[]>("model/fixtures.json");
  it("ships at least 20 fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));
  it("declares the feature layout the featurizer produces", () => {
    expect(MODEL.manifest.featureRows).toBe(639);
    expect(MODEL.manifest.labels).toHaveLength(14);
  });
  it.each(cases.map((c, i) => [i, c.text] as const))("case %i matches PyTorch: %s", (i) => {
    const c = cases[i]!;
    const f = featurize(c.text, c.schema);
    expect(f.rows).toEqual(c.rows);
    const logits = forwardCpu(MODEL, f);
    const outs = MODEL.manifest.labels.length + 1;
    expect(logits.length).toBe(c.tokens.length * outs);
    let worst = 0;
    for (let t = 0; t < c.tokens.length; t++) {
      for (let o = 0; o < outs; o++)
        worst = Math.max(worst, Math.abs(logits[t * outs + o]! - c.logits[t]![o]!));
    }
    expect(worst).toBeLessThan(1e-4);
  });
});
