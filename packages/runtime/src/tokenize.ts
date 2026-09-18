/**
 * Mechanical tokenizer shared by every gpu-utils model. Mirrors
 * tooling/python/gpu_utils_training/features.py exactly; parity is enforced by
 * test/fixtures/tokenize.json on both sides.
 */
export const CharClass = {
	Letter: 0,
	Digit: 1,
	Space: 2,
	Newline: 3,
	Other: 4,
} as const;
export const Shape = {
	Lower: 0,
	Upper: 1,
	Title: 2,
	Mixed: 3,
	Digits: 4,
	Space: 5,
	Newline: 6,
	Other: 7,
} as const;

export interface Token {
	text: string;
	/** UTF-16 code unit offsets, half-open. */
	start: number;
	end: number;
	cls: number;
	shape: number;
}

const LETTER = /^[\p{L}\p{M}]$/u;
const DIGIT = /^\p{Nd}$/u;
const LOWER = /^\p{Ll}+$/u;
const UPPER = /^\p{Lu}+$/u;
const TITLE = /^\p{Lu}\p{Ll}*$/u;

function charClass(ch: string): number {
	if (ch === "\n" || ch === "\r") return CharClass.Newline;
	if (ch === " " || ch === "\t") return CharClass.Space;
	if (LETTER.test(ch)) return CharClass.Letter;
	if (DIGIT.test(ch)) return CharClass.Digit;
	return CharClass.Other;
}

function shapeOf(text: string, cls: number): number {
	if (cls === CharClass.Space) return Shape.Space;
	if (cls === CharClass.Newline) return Shape.Newline;
	if (cls === CharClass.Digit) return Shape.Digits;
	if (cls === CharClass.Other) return Shape.Other;
	const letters = text.replace(/\p{M}/gu, "");
	if (LOWER.test(letters)) return Shape.Lower;
	if (UPPER.test(letters)) return Shape.Upper;
	if (TITLE.test(letters)) return Shape.Title;
	return Shape.Mixed;
}

export function tokenize(text: string): Token[] {
	const tokens: Token[] = [];
	const chars = Array.from(text); // code points
	let offset = 0;
	let i = 0;
	while (i < chars.length) {
		const ch = chars[i]!;
		const cls = charClass(ch);
		let j = i + 1;
		if (cls === CharClass.Newline) {
			if (ch === "\r" && chars[j] === "\n") j++;
		} else if (cls !== CharClass.Other) {
			while (j < chars.length && charClass(chars[j]!) === cls) j++;
		}
		const chunk = chars.slice(i, j).join("");
		tokens.push({
			text: chunk,
			start: offset,
			end: offset + chunk.length,
			cls,
			shape: shapeOf(chunk, cls),
		});
		offset += chunk.length;
		i = j;
	}
	return tokens;
}
