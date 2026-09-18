import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { dispatch, packParams } from "../src/gpu.ts";
import { MODEL } from "../src/model.ts";

interface Fixture {
  text: string;
  features: number[][];
  logits: number[];
}

const fixtures = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "../model/fixtures.json"), "utf8"),
) as Fixture[];

describe("gpu-tailwind parity", () => {
  it("featurizer matches the Python featurizer on every fixture", () => {
    for (const f of fixtures) expect(featurize(f.text).rows, f.text).toEqual(f.features);
  });

  it("CPU reference logits match PyTorch fixtures at 1e-4", () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(20);
    let worst = 0;
    for (const f of fixtures) {
      const got = forwardCpu(MODEL, featurize(f.text));
      expect(got.length).toBe(f.logits.length);
      for (let i = 0; i < got.length; i++)
        worst = Math.max(worst, Math.abs(got[i]! - f.logits[i]!));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it("GPU dispatch covers every entry point and packs tensor offsets", () => {
    const params = packParams(MODEL, 300);
    expect(params[0]).toBe(300);
    expect(params[1]).toBe(2);
    const shader = readFileSync(resolve(import.meta.dirname, "../src/shader.wgsl"), "utf8");
    for (const d of dispatch(MODEL, 300)) expect(shader).toContain(`fn ${d.entry}(`);
    // every entry point statically references every binding via touch()
    expect((shader.match(/touch\(\)/g) ?? []).length).toBeGreaterThanOrEqual(7);
  });
});
