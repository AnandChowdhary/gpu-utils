import { describe, expect, it } from "vitest";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { featurize } from "../src/features.ts";
import { MODEL } from "../src/model.ts";

interface Fixture {
	text: string;
	tokens: string[];
	features: number[][];
	span: number[];
	kind: number[];
}

const cases = fixtures as Fixture[];
const TOL = 1e-4;

/**
 * Fixtures are written by `pnpm export` (training/gpu_paste/export.py) from the
 * dequantized int6 weights. The CPU reference path must reproduce the PyTorch logits
 * within 1e-4; the WebGPU path is checked against the same fixtures by
 * training/tests/test_wgsl.py (lavapipe) and in the browser.
 */
describe("gpu-paste parity", () => {
	it("has at least 20 fixtures", () => {
		expect(cases.length).toBeGreaterThanOrEqual(20);
	});

	it("featurizer matches Python byte for byte", () => {
		for (const c of cases) {
			const f = featurize(c.text);
			expect(f.tokens.map((t) => t.text)).toEqual(c.tokens);
			expect(f.rows).toEqual(c.features);
		}
	});

	it("CPU logits match PyTorch fixtures at 1e-4", () => {
		let maxDiff = 0;
		for (const c of cases) {
			const f = featurize(c.text);
			const out = forwardCpu(MODEL, f.rows);
			expect(out.span.length).toBe(c.span.length);
			for (let i = 0; i < c.span.length; i++) {
				const d = Math.abs(out.span[i]! - c.span[i]!);
				maxDiff = Math.max(maxDiff, d);
				if (d > TOL)
					throw new Error(
						`span logit ${i} differs by ${d} for ${JSON.stringify(c.text)}`,
					);
			}
			if (c.kind.length) {
				for (let i = 0; i < c.kind.length; i++) {
					const d = Math.abs(out.kind[i]! - c.kind[i]!);
					maxDiff = Math.max(maxDiff, d);
					if (d > TOL)
						throw new Error(
							`kind logit ${i} differs by ${d} for ${JSON.stringify(c.text)}`,
						);
				}
			}
		}
		expect(maxDiff).toBeLessThanOrEqual(TOL);
	});
});
