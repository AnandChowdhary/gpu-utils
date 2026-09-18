import { type FeatureRows, tensor } from "@gpu-utils/runtime";
import type { Model } from "./model.ts";

/**
 * Reference forward pass in plain TypeScript. Returns [tokens, labels + 1] logits:
 * one column per role followed by the clause-boundary logit. Mirrors
 * training/gpu_view/model.py; src/shader.wgsl must match this to 1e-4.
 */
export function forwardCpu(model: Model, features: FeatureRows): Float32Array {
	const { manifest, weights } = model;
	const n = features.tokens.length;
	const H = Number(manifest.hidden);
	const G = Number(manifest.headGate);
	const C = Number(manifest.conv);
	const outs = manifest.labels.length + 1;
	const logits = new Float32Array(n * outs);
	if (n === 0) return logits;
	const W = (name: string) => tensor(weights, manifest, name);
	const embedding = W("embedding");
	const encoderBias = W("encoder_bias");
	const convolution = W("convolution");
	const gateW = W("gate_weight");
	const gateB = W("gate_bias");
	const candW = W("candidate_weight");
	const candB = W("candidate_bias");
	const combW = W("combine_weight");
	const combB = W("combine_bias");
	const globW = W("global_weight");
	const globB = W("global_bias");
	const hgW = W("head_gate_weight");
	const hgB = W("head_gate_bias");
	const hhW = W("head_hidden_weight");
	const hhB = W("head_hidden_bias");
	const outW = W("output_weight");
	const outB = W("output_bias");
	const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
	const half = (C - 1) >> 1;

	// 1. Summed sparse embeddings.
	const emb = new Float64Array(n * H);
	for (let t = 0; t < n; t++) {
		for (const row of features.rows[t]!) {
			for (let c = 0; c < H; c++)
				emb[t * H + c] = emb[t * H + c]! + embedding[row * H + c]!;
		}
	}
	// 2. Depthwise convolution + tanh.
	const enc = new Float64Array(n * H);
	for (let t = 0; t < n; t++) {
		for (let c = 0; c < H; c++) {
			let s = encoderBias[c]!;
			for (let k = 0; k < C; k++) {
				const src = t + k - half;
				if (src >= 0 && src < n)
					s += convolution[k * H + c]! * emb[src * H + c]!;
			}
			enc[t * H + c] = Math.tanh(s);
		}
	}
	// 3. Gate and candidate.
	const gate = new Float64Array(n * H);
	const cand = new Float64Array(n * H);
	for (let t = 0; t < n; t++) {
		for (let c = 0; c < H; c++) {
			let g = gateB[c]!;
			let k = candB[c]!;
			for (let i = 0; i < H; i++) {
				const v = enc[t * H + i]!;
				g += v * gateW[c * H + i]!;
				k += v * candW[c * H + i]!;
			}
			const gv = sigmoid(g);
			gate[t * H + c] = gv;
			cand[t * H + c] = (1 - gv) * Math.tanh(k);
		}
	}
	// 4. Affine scans, forward and backward: h[t] = a[t] * h[t-1] + b[t].
	const fwd = new Float64Array(n * H);
	const bwd = new Float64Array(n * H);
	for (let c = 0; c < H; c++) {
		let s = 0;
		for (let t = 0; t < n; t++) {
			s = gate[t * H + c]! * s + cand[t * H + c]!;
			fwd[t * H + c] = s;
		}
		s = 0;
		for (let t = n - 1; t >= 0; t--) {
			s = gate[t * H + c]! * s + cand[t * H + c]!;
			bwd[t * H + c] = s;
		}
	}
	// 5. Combine and mean-pool into a gated global context.
	const combined = new Float64Array(n * H);
	const pooled = new Float64Array(H);
	for (let t = 0; t < n; t++) {
		for (let c = 0; c < H; c++) {
			let s = combB[c]!;
			for (let i = 0; i < H; i++) {
				s +=
					fwd[t * H + i]! * combW[c * 2 * H + i]! +
					bwd[t * H + i]! * combW[c * 2 * H + H + i]!;
			}
			const v = Math.tanh(enc[t * H + c]! + s);
			combined[t * H + c] = v;
			pooled[c] = pooled[c]! + v / n;
		}
	}
	const context = new Float64Array(H);
	for (let c = 0; c < H; c++) {
		let s = globB[c]!;
		for (let i = 0; i < H; i++) s += pooled[i]! * globW[c * H + i]!;
		context[c] = sigmoid(s) * pooled[c]!;
	}
	// 6. Two-layer head per token.
	const joined = new Float64Array(2 * H + G);
	const hidden = new Float64Array(hhB.length);
	for (let c = 0; c < H; c++) joined[H + c] = context[c]!;
	for (let t = 0; t < n; t++) {
		for (let c = 0; c < H; c++) joined[c] = combined[t * H + c]!;
		for (let g = 0; g < G; g++) {
			let s = hgB[g]!;
			for (let i = 0; i < 2 * H; i++) s += joined[i]! * hgW[g * 2 * H + i]!;
			joined[2 * H + g] = sigmoid(s);
		}
		const width = 2 * H + G;
		for (let h = 0; h < hidden.length; h++) {
			let s = hhB[h]!;
			for (let i = 0; i < width; i++) s += joined[i]! * hhW[h * width + i]!;
			hidden[h] = Math.tanh(s);
		}
		for (let o = 0; o < outs; o++) {
			let s = outB[o]!;
			for (let h = 0; h < hidden.length; h++)
				s += hidden[h]! * outW[o * hidden.length + h]!;
			logits[t * outs + o] = s;
		}
	}
	return logits;
}
