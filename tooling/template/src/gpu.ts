import {
	createProgram,
	type FeatureRows,
	type Program,
} from "@gpu-utils/runtime";
import type { Model } from "./model.ts";
import shader from "./shader.wgsl";

let programPromise: Promise<Program> | undefined;

/** WebGPU forward pass. Must produce the same logits as forwardCpu (see test/parity). */
export async function forwardGpu(
	model: Model,
	features: FeatureRows,
): Promise<Float32Array> {
	programPromise ??= createProgram(shader, ["embed", "classify"]);
	const program = await programPromise;
	const n = features.tokens.length;
	const k = model.manifest.labels.length;
	const hidden = Number(model.manifest.hidden ?? 32);
	const featureWidth = features.rows[0]?.length ?? 0;

	const flat = new Uint32Array(n * featureWidth);
	for (const [i, row] of features.rows.entries())
		flat.set(row, i * featureWidth);

	const params = program.buffer(
		"params",
		new Uint32Array([n, hidden, k, featureWidth]),
		GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	);
	const featureBuf = program.buffer("features", flat);
	const weights = program.buffer("weights", model.weights);
	const state = program.buffer(
		"state",
		new Float32Array(Math.max(1, n * hidden)),
	);
	const logits = program.buffer(
		"logits",
		new Float32Array(Math.max(1, n * k)),
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	);

	const groups = Math.ceil(n / 64);
	const out = await program.run(
		[params, featureBuf, weights, state, logits],
		[
			{ entry: "embed", workgroups: [groups] },
			{ entry: "classify", workgroups: [groups] },
		],
		logits,
	);
	return out.subarray(0, n * k);
}
