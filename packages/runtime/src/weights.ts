/** Mirrors tooling/python/gpu_utils_training/quant.py. */
const ALPHABET = (() => {
	let s = "";
	for (let c = 35; c < 127 && s.length < 64; c++) {
		const ch = String.fromCharCode(c);
		if (!"\\\"'`".includes(ch)) s += ch;
	}
	return s;
})();
const LOOKUP = new Int8Array(128);
for (let i = 0; i < 64; i++) LOOKUP[ALPHABET.charCodeAt(i)] = i - 32;

export interface TensorEntry {
	name: string;
	offset: number;
	length: number;
	shape: number[];
	scale: number;
}

export interface ModelManifest {
	format: number;
	name: string;
	tensors: TensorEntry[];
	parameters: number;
	labels: string[];
	[key: string]: unknown;
}

/** Decodes the whole int6 weight string into a flat Float32Array (per-tensor scales applied). */
export function decodeInt6(
	encoded: string,
	tensors: TensorEntry[],
): Float32Array {
	const out = new Float32Array(encoded.length);
	for (const t of tensors) {
		for (let i = 0; i < t.length; i++) {
			out[t.offset + i] = LOOKUP[encoded.charCodeAt(t.offset + i)]! * t.scale;
		}
	}
	return out;
}

/** Returns a view of one named tensor within the flat decoded weights. */
export function tensor(
	weights: Float32Array,
	manifest: ModelManifest,
	name: string,
): Float32Array {
	const entry = manifest.tensors.find((t) => t.name === name);
	if (!entry) throw new Error(`unknown tensor ${name}`);
	return weights.subarray(entry.offset, entry.offset + entry.length);
}
