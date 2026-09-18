import { describe, expect, it } from "vitest";
import { hashToken } from "../src/hash.ts";
import { tokenize } from "../src/tokenize.ts";
import fixtures from "./fixtures/tokenize.json" with { type: "json" };

describe("tokenize", () => {
	for (const c of fixtures) {
		it(JSON.stringify(c.text), () => {
			expect(
				tokenize(c.text).map((t) => [t.text, t.start, t.end, t.cls, t.shape]),
			).toEqual(c.tokens);
		});
	}
	it("round-trips offsets", () => {
		const text = "a  b\n\nc\r\n";
		const tokens = tokenize(text);
		expect(tokens.map((t) => text.slice(t.start, t.end)).join("")).toBe(text);
	});
});

describe("hashToken", () => {
	for (const c of fixtures) {
		for (const [text, expected] of Object.entries(c.hashes)) {
			it(`${text} -> ${expected}`, () =>
				expect(hashToken(text, 1024)).toBe(expected));
		}
	}
	it("collapses digits and case", () => {
		expect(hashToken("ABC123", 4096)).toBe(hashToken("abc999", 4096));
	});
});
