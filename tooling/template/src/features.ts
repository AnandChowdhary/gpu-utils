import { type FeatureRows, hashToken, tokenize } from "@gpu-utils/runtime";

/**
 * CPU pre-pass: split text into tokens and emit sparse feature ids per token.
 * Must match training/__SNAKE__/features.py exactly. Parity is enforced by
 * test/parity.test.ts against fixtures exported from Python.
 */
export function featurize(text: string): FeatureRows {
	const tokens = tokenize(text);
	return {
		tokens,
		rows: tokens.map((t) => [
			hashToken(t.text, 1024),
			t.shape,
			Math.min(t.text.length, 15),
		]),
	};
}
