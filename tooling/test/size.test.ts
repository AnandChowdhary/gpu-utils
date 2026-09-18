import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { measure } from "../size.ts";

describe("size", () => {
	it("measures a built package and reads its budget", () => {
		const dir = mkdtempSync(join(tmpdir(), "gpu-utils-size-"));
		mkdirSync(join(dir, "dist"));
		writeFileSync(
			join(dir, "package.json"),
			JSON.stringify({ name: "x", gpuUtils: { sizeBudget: 100 } }),
		);
		writeFileSync(
			join(dir, "dist/index.js"),
			Array.from({ length: 20 }, (_, i) => `export const v${i} = ${i};`).join(
				"\n",
			),
		);
		const r = measure(dir)!;
		expect(r.name).toBe("x");
		expect(r.budget).toBe(100);
		expect(r.bytes).toBeGreaterThan(0);
	});
	it("skips private packages", () => {
		const dir = mkdtempSync(join(tmpdir(), "gpu-utils-size-"));
		writeFileSync(
			join(dir, "package.json"),
			JSON.stringify({ name: "y", private: true }),
		);
		expect(measure(dir)).toBeNull();
	});
});
