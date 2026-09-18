import type { Token } from "./tokenize.ts";

/** "auto" picks WebGPU for batched/large inputs and CPU otherwise; "cpu" is the reference path. */
export type Backend = "auto" | "webgpu" | "cpu";

export interface FeatureRows {
	tokens: Token[];
	/** One row of sparse feature ids per token. */
	rows: number[][];
}
