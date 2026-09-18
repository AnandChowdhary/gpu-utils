import { describe, expect, it } from "vitest";
import { compile, type Role } from "../src/decode.ts";
import type { Schema } from "../src/match.ts";
import { modelTokens } from "../src/match.ts";
import { canon, read, strip } from "./helpers.ts";

type Gold = {
	text: string;
	schema: Schema;
	now: string;
	roles: Role[];
	boundaries: number[];
	spec: unknown;
};

/**
 * The compiler must rebuild the generator's gold spec from the gold roles for every
 * generated example. Anything below 100% is a compiler or generator bug, not a model
 * result, so this is the invariant that makes the model metrics meaningful.
 */
describe("compiler round-trips gold labels", () => {
	for (const file of ["eval/heldout.json", "eval/indomain.json"]) {
		it(`${file}: 100% exact`, () => {
			const cases = read<Gold[]>(file);
			const failures: string[] = [];
			for (const c of cases) {
				const tokens = modelTokens(c.text);
				const spec = compile(
					c.text,
					tokens,
					c.roles,
					c.boundaries.map(Boolean),
					c.schema,
					{
						now: c.now,
					},
				);
				if (canon(strip(spec)) !== canon(c.spec)) {
					failures.push(
						`${c.text}\n   got  ${canon(strip(spec))}\n   want ${canon(c.spec)}\n   diag ${JSON.stringify(spec.diagnostics)}`,
					);
				}
			}
			if (failures.length)
				console.log(
					`${failures.length}/${cases.length} mismatches\n${failures.slice(0, 12).join("\n")}`,
				);
			expect(failures).toHaveLength(0);
		});
	}
});
