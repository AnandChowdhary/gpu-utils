/** Row-wise argmax over a [n, k] logits matrix. */
export function argmax(logits: Float32Array, n: number, k: number): Int32Array {
	const out = new Int32Array(n);
	for (let i = 0; i < n; i++) {
		let best = 0;
		let bestValue = -Infinity;
		for (let j = 0; j < k; j++) {
			const v = logits[i * k + j]!;
			if (v > bestValue) {
				bestValue = v;
				best = j;
			}
		}
		out[i] = best;
	}
	return out;
}

/**
 * Viterbi decode over [n, k] emission scores with a [k, k] transition matrix
 * (transitions[from * k + to]); use -Infinity to forbid a transition (e.g. O → I-X).
 */
export function viterbi(
	emissions: Float32Array,
	n: number,
	k: number,
	transitions: Float32Array,
): Int32Array {
	if (n === 0) return new Int32Array(0);
	const score = new Float64Array(n * k);
	const back = new Int32Array(n * k);
	for (let j = 0; j < k; j++) score[j] = emissions[j]!;
	for (let i = 1; i < n; i++) {
		for (let to = 0; to < k; to++) {
			let best = -Infinity;
			let arg = 0;
			for (let from = 0; from < k; from++) {
				const s = score[(i - 1) * k + from]! + transitions[from * k + to]!;
				if (s > best) {
					best = s;
					arg = from;
				}
			}
			score[i * k + to] = best + emissions[i * k + to]!;
			back[i * k + to] = arg;
		}
	}
	const path = new Int32Array(n);
	let last = 0;
	for (let j = 1; j < k; j++)
		if (score[(n - 1) * k + j]! > score[(n - 1) * k + last]!) last = j;
	path[n - 1] = last;
	for (let i = n - 1; i > 0; i--) path[i - 1] = back[i * k + path[i]!]!;
	return path;
}
