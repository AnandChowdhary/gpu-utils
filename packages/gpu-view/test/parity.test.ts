import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { FEATURE_ROWS, featurize, SLOTS } from "../src/features.ts";
import type { Schema } from "../src/match.ts";
import { MODEL } from "../src/model.ts";

/**
 * model/fixtures.json is written by `pnpm export` (gpu_utils_training.export.export_package)
 * in the canonical format { cases: [{ input: { text, schema }, rows, logits, pooled }] },
 * computed by the Python scan-family reference on the decoded int6 weights. The featurizer
 * must reproduce `rows` byte for byte and the CPU forward pass must reproduce `logits` at
 * 1e-4. The WGSL path is checked against the same fixtures by training/tests/test_wgsl.py.
 */
type Case = { input: { text: string; schema: Schema }; rows: number[][]; logits: number[][] };
const cases = (fixtures as { cases: Case[] }).cases;

describe("gpu-view parity", () => {
  it("ships at least 20 fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));
  it("declares the feature layout the featurizer produces", () => {
    expect(MODEL.manifest.family).toBe("scan");
    expect(MODEL.manifest.featureRows).toBe(FEATURE_ROWS);
    expect(MODEL.manifest.slots).toBe(SLOTS);
    expect(MODEL.manifest.labels).toHaveLength(14);
    expect(MODEL.manifest.tags).toBe(15);
  });
  it.each(cases.map((c, i) => [i, c.input.text] as const))("case %i matches PyTorch: %s", (i) => {
    const c = cases[i]!;
    const f = featurize(c.input.text, c.input.schema);
    expect(f.rows).toEqual(c.rows);
    const logits = forwardCpu(MODEL, f);
    const outs = MODEL.manifest.tags;
    expect(logits.length).toBe(c.rows.length * outs);
    let worst = 0;
    for (let t = 0; t < c.rows.length; t++) {
      for (let o = 0; o < outs; o++) worst = Math.max(worst, Math.abs(logits[t * outs + o]! - c.logits[t]![o]!));
    }
    expect(worst).toBeLessThan(1e-4);
  });
});
