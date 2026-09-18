import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ViewSpec } from "../src/decode.ts";

export const read = <T>(rel: string): T =>
	JSON.parse(
		readFileSync(resolve(import.meta.dirname, "..", rel), "utf8"),
	) as T;

/** Drops diagnostics/tokens (and optionally spans) so specs can be compared to gold. */
export function strip(spec: ViewSpec, withSpans = true): unknown {
	const noSpan = <T extends { span?: unknown }>(x: T) => {
		const { span: _span, ...rest } = x;
		return withSpans ? x : rest;
	};
	const out: Record<string, unknown> = {
		filters: spec.filters.map(noSpan),
		sort: spec.sort.map(noSpan),
		groupBy: spec.groupBy.map(noSpan),
		aggregate: spec.aggregate.map(noSpan),
	};
	if (spec.limit !== undefined) out.limit = spec.limit;
	if (spec.chart !== undefined) out.chart = spec.chart;
	if (spec.granularity !== undefined) out.granularity = spec.granularity;
	return out;
}

export const canon = (x: unknown) => JSON.stringify(sortKeys(x));
function sortKeys(x: unknown): unknown {
	if (Array.isArray(x)) return x.map(sortKeys);
	if (x && typeof x === "object")
		return Object.fromEntries(
			Object.keys(x)
				.sort()
				.map((k) => [k, sortKeys((x as Record<string, unknown>)[k])]),
		);
	return x;
}
