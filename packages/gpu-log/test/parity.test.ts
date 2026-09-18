import { tokenize } from "@gpu-utils/runtime";
import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { type Batch, forwardCpu } from "../src/cpu.ts";
import { FEATURE_COUNT, featurize, writeTokenFeatures } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

interface Fixture {
  text: string;
  features: number[][];
  logits: number[][];
}

/**
 * Fixtures are written by `pnpm export` (training/gpu_log/export.py): per case the line, the
 * Python feature rows and the float32 logits of the numpy reference forward computed from the
 * dequantized int6 tensors. The TypeScript featurizer must match the rows exactly and the CPU
 * forward pass must reproduce the logits within 1e-4. The WGSL path is checked against the CPU
 * path in test/gpu.test.ts when a WebGPU device is available.
 */
describe("gpu-log parity", () => {
  const cases = fixtures as Fixture[];
  it("has enough fixtures", () => expect(cases.length).toBeGreaterThanOrEqual(20));

  for (const c of cases) {
    it(`features: ${JSON.stringify(c.text.slice(0, 40))}`, () => {
      expect(featurize(c.text).rows).toEqual(c.features);
    });
    it(`logits: ${JSON.stringify(c.text.slice(0, 40))}`, () => {
      const tokens = tokenize(c.text);
      const n = tokens.length;
      const features = new Uint32Array(n * FEATURE_COUNT);
      for (let i = 0; i < n; i++) writeTokenFeatures(tokens, i, features, i * FEATURE_COUNT);
      const batch: Batch = { n, features, lineId: new Uint32Array(n), offsets: [0, n] };
      const out = forwardCpu(MODEL, batch);
      const W = MODEL.tags + MODEL.kinds;
      expect(out.length).toBe(n * W);
      let maxDiff = 0;
      for (let p = 0; p < n; p++) {
        for (let j = 0; j < W; j++)
          maxDiff = Math.max(maxDiff, Math.abs(out[p * W + j]! - c.logits[p]![j]!));
      }
      expect(maxDiff).toBeLessThan(1e-4);
    });
  }

  it("packing several lines in one batch gives the same logits as one line per batch", () => {
    const texts = cases
      .filter((c) => c.text.length > 0)
      .slice(0, 6)
      .map((c) => c.text);
    const per = texts.map((t) => {
      const tokens = tokenize(t);
      const features = new Uint32Array(tokens.length * FEATURE_COUNT);
      for (let i = 0; i < tokens.length; i++)
        writeTokenFeatures(tokens, i, features, i * FEATURE_COUNT);
      return forwardCpu(MODEL, {
        n: tokens.length,
        features,
        lineId: new Uint32Array(tokens.length),
        offsets: [0, tokens.length],
      });
    });
    const all = texts.map((t) => tokenize(t));
    const n = all.reduce((s, t) => s + t.length, 0);
    const features = new Uint32Array(n * FEATURE_COUNT);
    const lineId = new Uint32Array(n);
    const offsets = [0];
    let p = 0;
    for (const [li, tokens] of all.entries()) {
      for (let i = 0; i < tokens.length; i++, p++) {
        writeTokenFeatures(tokens, i, features, p * FEATURE_COUNT);
        lineId[p] = li;
      }
      offsets.push(p);
    }
    const packed = forwardCpu(MODEL, { n, features, lineId, offsets });
    const W = MODEL.tags + MODEL.kinds;
    for (const [li, single] of per.entries()) {
      const slice = packed.subarray(offsets[li]! * W, offsets[li + 1]! * W);
      for (let j = 0; j < single.length; j++)
        expect(Math.abs(slice[j]! - single[j]!)).toBeLessThan(1e-5);
    }
  });
});
