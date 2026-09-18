import { decodeInt6, type ModelManifest, tensor } from "@gpu-utils/runtime";
import manifest from "../model/manifest.json" with { type: "json" };
import encoded from "../model/weights.txt";

export interface CiteManifest extends ModelManifest {
	embed: number;
	hidden: number;
	head: number;
	tableRows: number;
	featureWidth: number;
	roles: string[];
	types: string[];
	nameparts: string[];
}

export interface Model {
	manifest: CiteManifest;
	/** Flat float32 weights; use `tensor(weights, manifest, name)` for a named view. */
	weights: Float32Array;
	/** Named views into `weights`, decoded once. */
	t: Record<string, Float32Array>;
}

const TENSOR_NAMES = [
	"emb",
	"s1f_wa",
	"s1f_ba",
	"s1f_wb",
	"s1f_bb",
	"s1b_wa",
	"s1b_ba",
	"s1b_wb",
	"s1b_bb",
	"conv_w",
	"conv_b",
	"s2f_wa",
	"s2f_ba",
	"s2f_wb",
	"s2f_bb",
	"s2b_wa",
	"s2b_ba",
	"s2b_wb",
	"s2b_bb",
	"w1",
	"b1",
	"wt",
	"bt",
	"wp",
	"bp",
	"wc",
	"bc",
	"wd",
	"bd",
	"trans",
] as const;

export function loadModel(m: CiteManifest, weightText: string): Model {
	const weights = decodeInt6(weightText, m.tensors);
	const t: Record<string, Float32Array> = {};
	for (const name of TENSOR_NAMES) t[name] = tensor(weights, m, name);
	return { manifest: m, weights, t };
}

/** Trained weights, decoded once at import time. Written by training/gpu_cite/export.py. */
export const MODEL: Model = loadModel(
	manifest as unknown as CiteManifest,
	encoded,
);
