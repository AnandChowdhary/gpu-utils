import {
	createProgram,
	type FeatureRows,
	type Program,
} from "@gpu-utils/runtime";
import type { Logits } from "./cpu.ts";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

const ENTRIES = [
	"embed",
	"gates1",
	"scan1",
	"conv",
	"gates2",
	"scan2",
	"pool",
	"head_hidden",
	"head_out",
	"type_hidden",
	"type_out",
];

/** Order must match the M_* constants in shader.wgsl. */
const TENSOR_ORDER = [
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
];
const STATE_SLOTS = 9; // e, g1, h1, y, g2, h2, pool, g, c
const META_SEQ = TENSOR_ORDER.length + STATE_SLOTS;

let programPromise: Promise<Program> | undefined;
const weightBuffers = new WeakMap<Model, GPUBuffer>();

/**
 * WebGPU forward pass over a batch of references (one flat token stream). Produces the same
 * logits as forwardCpu for every sequence; see test/parity.test.ts.
 */
export async function forwardGpuBatch(
	model: Model,
	batch: FeatureRows[],
): Promise<Logits[]> {
	programPromise ??= createProgram(shader, ENTRIES);
	const program = await programPromise;
	const m = model.manifest;
	const E = m.embed;
	const H = m.hidden;
	const C = 2 * H;
	const HEAD = m.head;
	const K = m.labels.length;
	const P = m.nameparts.length;
	const T = m.types.length;
	const W = m.featureWidth;
	const S = batch.length;
	const lengths = batch.map((f) => f.tokens.length);
	const N = lengths.reduce((a, b) => a + b, 0);
	if (S === 0) return [];
	if (N === 0) {
		return batch.map(() => ({
			n: 0,
			tags: new Float32Array(0),
			parts: new Float32Array(0),
			type: new Float32Array(T),
		}));
	}

	// Flat features and meta (tensor offsets, state offsets, sequence table, token→sequence).
	const features = new Uint32Array(Math.max(1, N * W));
	const meta = new Uint32Array(META_SEQ + 2 * S + N);
	const tokSeq = META_SEQ + 2 * S;
	let cursor = 0;
	for (let s = 0; s < S; s++) {
		meta[META_SEQ + 2 * s] = cursor;
		meta[META_SEQ + 2 * s + 1] = lengths[s]!;
		for (const row of batch[s]!.rows) {
			features.set(row, cursor * W);
			meta[tokSeq + cursor] = s;
			cursor++;
		}
	}
	for (const [i, name] of TENSOR_ORDER.entries()) {
		const entry = m.tensors.find((t) => t.name === name);
		if (!entry) throw new Error(`gpu-cite: manifest is missing tensor ${name}`);
		meta[i] = entry.offset;
	}
	const stateSizes = [
		N * E,
		N * C * 2,
		N * C,
		N * C,
		N * C * 2,
		N * C,
		S * 2 * C,
		N * HEAD,
		S * H,
	];
	let stateTotal = 0;
	for (const [i, size] of stateSizes.entries()) {
		meta[TENSOR_ORDER.length + i] = stateTotal;
		stateTotal += size;
	}
	const logitTotal = N * K + N * P + S * T;

	const params = program.buffer(
		"params",
		new Uint32Array([N, S, E, H, HEAD, K, P, T, W, 0, 0, 0]),
		GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	);
	const metaBuf = program.buffer("meta", meta);
	const featureBuf = program.buffer("features", features);
	let weights = weightBuffers.get(model);
	if (!weights) {
		weights = program.buffer("weights", model.weights);
		weightBuffers.set(model, weights);
	}
	const state = program.buffer(
		"state",
		new Float32Array(Math.max(1, stateTotal)),
	);
	const logits = program.buffer(
		"logits",
		new Float32Array(Math.max(1, logitTotal)),
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	);

	const groups = (count: number): [number] => [
		Math.max(1, Math.ceil(count / 64)),
	];
	const out = await program.run(
		[params, metaBuf, featureBuf, weights, state, logits],
		[
			{ entry: "embed", workgroups: groups(N * E) },
			{ entry: "gates1", workgroups: groups(N * C) },
			{ entry: "scan1", workgroups: [S * C] },
			{ entry: "conv", workgroups: groups(N * C) },
			{ entry: "gates2", workgroups: groups(N * C) },
			{ entry: "scan2", workgroups: [S * C] },
			{ entry: "pool", workgroups: groups(S * C) },
			{ entry: "head_hidden", workgroups: groups(N * HEAD) },
			{ entry: "head_out", workgroups: groups(N * (K + P)) },
			{ entry: "type_hidden", workgroups: groups(S * H) },
			{ entry: "type_out", workgroups: groups(S * T) },
		],
		logits,
	);

	const results: Logits[] = [];
	cursor = 0;
	for (let s = 0; s < S; s++) {
		const n = lengths[s]!;
		results.push({
			n,
			tags: out.slice(cursor * K, (cursor + n) * K),
			parts: out.slice(N * K + cursor * P, N * K + (cursor + n) * P),
			type: out.slice(N * K + N * P + s * T, N * K + N * P + (s + 1) * T),
		});
		cursor += n;
	}
	return results;
}

/** Single-reference convenience wrapper around forwardGpuBatch. */
export async function forwardGpu(
	model: Model,
	features: FeatureRows,
): Promise<Logits> {
	const [logits] = await forwardGpuBatch(model, [features]);
	return logits!;
}
