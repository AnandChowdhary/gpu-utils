/**
 * gpu-cite: parse freeform citation and reference strings into structured bibliographic fields.
 *
 * Public API. The tokenizer/featurizer lives in ./features.ts, the reference forward pass in
 * ./cpu.ts, the WebGPU forward pass in ./gpu.ts and the tag→record compiler in ./decode.ts.
 * Both forward passes produce identical logits (see test/parity.test.ts).
 */
import { type Backend, type FeatureRows, hasWebGPU } from "@gpu-utils/runtime";
import { forwardCpu, type Logits } from "./cpu.ts";
import { type CiteRecord, decode } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpuBatch } from "./gpu.ts";
import { MODEL } from "./model.ts";

export type {
	CiteDiagnostics,
	CiteField,
	CiteRecord,
	CiteType,
	Person,
} from "./decode.ts";
export { featurize } from "./features.ts";

export interface CiteOptions {
	/** "auto" uses WebGPU for batched/large inputs when available and the CPU reference otherwise. */
	backend?: Backend;
}

/** Inputs shorter than this run on the CPU under "auto": GPU readback latency dominates. */
const GPU_MIN_TOKENS = 256;
/** Sequences per GPU dispatch (bounded by workgroup-count limits). */
const GPU_BATCH = 512;

export class CiteInputError extends Error {
	override name = "CiteInputError";
}

/** Newlines and carriage returns become spaces so offsets are preserved. */
function flatten(text: string): string {
	return text.replace(/[\r\n]/g, " ");
}

async function forward(
	features: FeatureRows[],
	backend: Backend,
): Promise<Logits[]> {
	const total = features.reduce((s, f) => s + f.tokens.length, 0);
	const useGpu =
		backend === "webgpu" ||
		(backend === "auto" && hasWebGPU() && total >= GPU_MIN_TOKENS);
	if (!useGpu) return features.map((f) => forwardCpu(MODEL, f));
	const out: Logits[] = [];
	for (let i = 0; i < features.length; i += GPU_BATCH) {
		out.push(
			...(await forwardGpuBatch(MODEL, features.slice(i, i + GPU_BATCH))),
		);
	}
	return out;
}

/** Parses one reference string (any citation style). Newlines inside it are treated as spaces. */
export async function parse(
	text: string,
	options: CiteOptions = {},
): Promise<CiteRecord> {
	if (typeof text !== "string")
		throw new CiteInputError("parse() expects a string");
	const flat = flatten(text);
	const features = featurize(flat);
	const [logits] = await forward([features], options.backend ?? "auto");
	return decode(MODEL, features, logits!, flat, 0);
}

/**
 * Parses many references separated by newlines. Blank lines are skipped; every record's
 * `range` and `spans` are offsets into the original text.
 */
export async function parseMany(
	text: string,
	options: CiteOptions = {},
): Promise<CiteRecord[]> {
	if (typeof text !== "string")
		throw new CiteInputError("parseMany() expects a string");
	const lines: { text: string; start: number }[] = [];
	for (const m of text.matchAll(/[^\r\n]+/g)) {
		if (m[0].trim().length > 0) lines.push({ text: m[0], start: m.index });
	}
	const features = lines.map((l) => featurize(l.text));
	const logits = await forward(features, options.backend ?? "auto");
	return lines.map((l, i) =>
		decode(MODEL, features[i]!, logits[i]!, l.text, l.start),
	);
}
