import {
	CharClass,
	type FeatureRows,
	hashToken,
	type Token,
	tokenize,
} from "@gpu-utils/runtime";

/**
 * CPU pre-pass: split text into tokens and emit FEATURE_COUNT sparse feature ids per token,
 * all indexing one flat embedding table. Must match training/gpu_paste/features.py exactly;
 * parity is enforced by test/parity.test.ts against fixtures exported from Python.
 */
const WORD_BUCKETS = 1024;
const SKELETON_BUCKETS = 256;
const SHAPE_ROWS = 8;
const AFFIX_BUCKETS = 128;
const LENGTH_ROWS = 16;
const COLUMN_ROWS = 6;
const LINE_ROWS = 5;
const FROM_END_ROWS = 3;

const BASE_SKELETON = WORD_BUCKETS;
const BASE_SHAPE = BASE_SKELETON + SKELETON_BUCKETS;
const BASE_PREFIX = BASE_SHAPE + SHAPE_ROWS;
const BASE_SUFFIX = BASE_PREFIX + AFFIX_BUCKETS;
const BASE_LENGTH = BASE_SUFFIX + AFFIX_BUCKETS;
const BASE_COLUMN = BASE_LENGTH + LENGTH_ROWS;
const BASE_LINE = BASE_COLUMN + COLUMN_ROWS;
const BASE_FROM_END = BASE_LINE + LINE_ROWS;
const BASE_BIAS = BASE_FROM_END + FROM_END_ROWS;
export const TOTAL_ROWS = BASE_BIAS + 1;
export const FEATURE_COUNT = 10;

const VOWEL = /[aeiou]/g;

/** First letter plus the remaining non-vowel letters (lowercased); non-letter runs unchanged. */
function skeleton(text: string, cls: number): string {
	if (cls !== CharClass.Letter) return text;
	const lowered = text.toLowerCase();
	const chars = Array.from(lowered);
	return chars[0]! + chars.slice(1).join("").replace(VOWEL, "");
}

export function featurizeTokens(tokens: Token[]): number[][] {
	let totalLines = 1;
	for (const t of tokens) if (t.cls === CharClass.Newline) totalLines++;
	const rows: number[][] = [];
	let line = 0;
	let column = 0;
	for (const t of tokens) {
		const chars = Array.from(t.text);
		rows.push([
			hashToken(t.text, WORD_BUCKETS),
			BASE_SKELETON + hashToken(skeleton(t.text, t.cls), SKELETON_BUCKETS),
			BASE_SHAPE + t.shape,
			BASE_PREFIX + hashToken(chars.slice(0, 2).join(""), AFFIX_BUCKETS),
			BASE_SUFFIX + hashToken(chars.slice(-2).join(""), AFFIX_BUCKETS),
			BASE_LENGTH + Math.min(t.end - t.start, LENGTH_ROWS - 1),
			BASE_COLUMN + Math.min(column, COLUMN_ROWS - 1),
			BASE_LINE + Math.min(line, LINE_ROWS - 1),
			BASE_FROM_END + Math.min(totalLines - 1 - line, FROM_END_ROWS - 1),
			BASE_BIAS,
		]);
		if (t.cls === CharClass.Newline) {
			line++;
			column = 0;
		} else {
			column++;
		}
	}
	return rows;
}

export function featurize(text: string): FeatureRows {
	const tokens = tokenize(text);
	return { tokens, rows: featurizeTokens(tokens) };
}
