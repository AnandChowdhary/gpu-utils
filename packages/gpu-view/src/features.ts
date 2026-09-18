/**
 * Sparse feature rows per model token. Must match training/gpu_view/features.py byte
 * for byte (fixtures in test/fixtures/features.json).
 *
 * The model never sees a field name. Schema membership arrives as anonymous rows:
 * "matched a field of kind K (begin/inside, exact/stem/prefix/typo, via an alias)",
 * "matched an enum value (unique owner? owned by the nearest preceding/following
 * field?)", and the kind of / distance to the nearest field match on either side.
 * The only exact word identities are the closed task lexicon (lexicon.ts); everything
 * else is a hashed bucket (word + consonant skeleton), shape and length. Whitespace
 * tokens are dropped before the model.
 */
import {
	CharClass,
	type FeatureRows,
	hashToken,
	Shape,
} from "@gpu-utils/runtime";
import { KEYWORD_COUNT, KEYWORD_ID } from "./lexicon.ts";
import {
	enumEntries,
	type FieldKind,
	fieldEntries,
	matchSpans,
	modelTokens,
	type Schema,
	type Span,
} from "./match.ts";

const KIND_ID: Record<FieldKind, number> = {
	text: 1,
	number: 2,
	date: 3,
	enum: 4,
	boolean: 5,
};
const LENGTH_BUCKETS = [1, 2, 3, 4, 6, 8, 12];
const WORD_BUCKETS = 128;
const SKEL_BUCKETS = 64;
const DIST_ROWS = 6;

export const BLOCKS: [string, number][] = [
	["shape", 8],
	["length", 8],
	["word", WORD_BUCKETS],
	["skeleton", SKEL_BUCKETS],
	["keyword", KEYWORD_COUNT + 1],
	["flags", 10],
	["field_kind", 6],
	["field_pos", 3],
	["quality", 5],
	["alias", 2],
	["enum_any", 2],
	["enum_pos", 3],
	["enum_unique", 2],
	["enum_prev", 2],
	["enum_next", 2],
	["prev_kind", 6],
	["prev_dist", DIST_ROWS],
	["next_kind", 6],
	["next_dist", DIST_ROWS],
	["position", 4],
];
export const OFFSET: Record<string, number> = {};
let cursor = 0;
for (const [name, size] of BLOCKS) {
	OFFSET[name] = cursor;
	cursor += size;
}
export const FEATURE_ROWS = cursor;
export const PADDING_ROW = FEATURE_ROWS;
export const SLOTS = 28;

const SUFFIXES = new Set(["k", "m", "b", "bn", "mm", "%"]);
const DIGIT = /\p{Nd}/u;

function lengthBucket(n: number): number {
	for (let i = 0; i < LENGTH_BUCKETS.length; i++)
		if (n <= LENGTH_BUCKETS[i]!) return i;
	return LENGTH_BUCKETS.length;
}

function skeleton(text: string): string {
	const s = text.toLowerCase().replace(/[aeiou]/g, "");
	return s || text.toLowerCase();
}

const distBucket = (d: number | null) =>
	d === null ? DIST_ROWS - 1 : Math.min(d, DIST_ROWS - 2);

export interface ViewFeatures extends FeatureRows {
	fieldSpans: Span[];
	enumSpans: Span[];
}

export function featurize(text: string, schema: Schema): ViewFeatures {
	const tokens = modelTokens(text);
	const n = tokens.length;
	const fields = matchSpans(tokens, fieldEntries(schema));
	const enums = matchSpans(tokens, enumEntries(schema));
	const rows: number[][] = [];
	if (n === 0) return { tokens, rows, fieldSpans: fields, enumSpans: enums };

	const fieldAt: (Span | null)[] = new Array(n).fill(null);
	for (const s of fields) for (let i = s.start; i < s.end; i++) fieldAt[i] = s;
	const enumAt: (Span | null)[] = new Array(n).fill(null);
	for (const s of enums) for (let i = s.start; i < s.end; i++) enumAt[i] = s;

	const prevSpan: (Span | null)[] = new Array(n).fill(null);
	const nextSpan: (Span | null)[] = new Array(n).fill(null);
	let last: Span | null = null;
	for (let i = 0; i < n; i++) {
		const own = fieldAt[i]!;
		prevSpan[i] = own === null || last !== own ? last : before(fields, own);
		if (own !== null) last = own;
	}
	let upcoming: Span | null = null;
	for (let i = n - 1; i >= 0; i--) {
		const own = fieldAt[i]!;
		nextSpan[i] =
			own === null || upcoming !== own ? upcoming : after(fields, own);
		if (own !== null) upcoming = own;
	}

	for (let i = 0; i < n; i++) {
		const tok = tokens[i]!;
		const low = tok.text.toLowerCase();
		const r: number[] = [];
		const add = (block: string, v: number) => r.push(OFFSET[block]! + v);

		add("shape", tok.shape);
		add("length", lengthBucket(Array.from(tok.text).length)); // code points, like Python len()
		add("word", hashToken(tok.text, WORD_BUCKETS));
		add("skeleton", hashToken(skeleton(tok.text), SKEL_BUCKETS));
		add("keyword", KEYWORD_ID.get(low) ?? KEYWORD_COUNT);

		if (DIGIT.test(tok.text)) add("flags", 0);
		if (tok.cls === CharClass.Digit) add("flags", 1);
		if (tok.cls === CharClass.Other) add("flags", 2);
		if (tok.shape === Shape.Title || tok.shape === Shape.Upper) add("flags", 3);
		if (i === 0) add("flags", 4);
		if (i === n - 1) add("flags", 5);
		const prevDigit = i > 0 && tokens[i - 1]!.cls === CharClass.Digit;
		if (prevDigit) add("flags", 6);
		if (i + 1 < n && tokens[i + 1]!.cls === CharClass.Digit) add("flags", 7);
		if (
			tok.cls === CharClass.Digit &&
			tok.text.length === 4 &&
			(tok.text.startsWith("19") || tok.text.startsWith("20"))
		)
			add("flags", 8);
		if (prevDigit && SUFFIXES.has(low)) add("flags", 9);

		const f = fieldAt[i]!;
		add("field_kind", f ? KIND_ID[f.kind] : 0);
		add("field_pos", f === null ? 0 : i === f.start ? 1 : 2);
		add("quality", f ? f.quality + 1 : 0);
		add("alias", f?.alias ? 1 : 0);

		const e = enumAt[i]!;
		const p = prevSpan[i]!;
		const q = nextSpan[i]!;
		add("enum_any", e ? 1 : 0);
		add("enum_pos", e === null ? 0 : i === e.start ? 1 : 2);
		add("enum_unique", e && e.owners.length === 1 ? 1 : 0);
		add("enum_prev", e && p !== null && e.owners.includes(p.field) ? 1 : 0);
		add("enum_next", e && q !== null && e.owners.includes(q.field) ? 1 : 0);

		add("prev_kind", p ? KIND_ID[p.kind] : 0);
		add("prev_dist", distBucket(p ? i - p.end : null));
		add("next_kind", q ? KIND_ID[q.kind] : 0);
		add("next_dist", distBucket(q ? q.start - i - 1 : null));
		add("position", Math.min(3, Math.floor((4 * i) / n)));
		rows.push(r);
	}
	return { tokens, rows, fieldSpans: fields, enumSpans: enums };
}

function before(spans: Span[], own: Span): Span | null {
	let best: Span | null = null;
	for (const s of spans) if (s.end <= own.start) best = s;
	return best;
}

function after(spans: Span[], own: Span): Span | null {
	for (const s of spans) if (s.start >= own.end) return s;
	return null;
}
