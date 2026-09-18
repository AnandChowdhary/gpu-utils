/**
 * Tolerant surface-form matching of query tokens against a schema.
 * Mirrors training/gpu_view/match.py exactly (fixtures in test/fixtures/match.json).
 *
 * Matches n-grams (1..3 words) of letter/digit tokens, allowing "-" and "_" between
 * words. Quality tiers, best first: exact, stem (plural/inflection), prefix (>= 4
 * chars, unique), typo (>= 5 chars, one edit, unique). Multi-word spans use exact and
 * stem only. Both the featurizer and the compiler call this, so they always agree.
 */
import { CharClass, type Token, tokenize } from "@gpu-utils/runtime";

export type FieldKind = "text" | "number" | "date" | "enum" | "boolean";

export interface SchemaField {
	name: string;
	kind: FieldKind;
	aliases?: string[];
	values?: string[];
}

export interface Schema {
	fields: SchemaField[];
}

export const EXACT = 0;
export const STEM = 1;
export const PREFIX = 2;
export const TYPO = 3;
const MIN_PREFIX = 4;
const MIN_TYPO = 5;
const MAX_WORDS = 3;
const CONNECTORS = new Set(["-", "_"]);

export interface Entry {
	words: string[];
	field: number;
	kind: FieldKind;
	alias: boolean;
	value: number;
}

export interface Span {
	start: number;
	end: number;
	field: number;
	kind: FieldKind;
	quality: number;
	alias: boolean;
	value: number;
	owners: number[];
}

const isWord = (t: Token) =>
	t.cls === CharClass.Letter || t.cls === CharClass.Digit;

/** Lower-cased letter/digit runs of a surface form; camelCase and snake_case split. */
export function wordsOf(surface: string): string[] {
	const text = surface.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
	return tokenize(text)
		.filter(isWord)
		.map((t) => t.text.toLowerCase());
}

export function stem(word: string): string {
	const n = word.length;
	if (n >= 5 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
	if (n >= 5 && word.endsWith("sses")) return word.slice(0, -2);
	if (n >= 5 && word.endsWith("es") && "sxz".includes(word[n - 3]!))
		return word.slice(0, -2);
	if (n >= 6 && (word.endsWith("shes") || word.endsWith("ches")))
		return word.slice(0, -2);
	if (n >= 4 && word.endsWith("s") && !word.endsWith("ss"))
		return word.slice(0, -1);
	return word;
}

export function withinOneEdit(a: string, b: string): boolean {
	if (a === b || Math.abs(a.length - b.length) > 1) return false;
	if (a.length === b.length) {
		let diff = 0;
		for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
		return diff === 1;
	}
	const [short, long] = a.length < b.length ? [a, b] : [b, a];
	let i = 0;
	while (i < short.length && short[i] === long[i]) i++;
	return short.slice(i) === long.slice(i + 1);
}

export function fieldEntries(schema: Schema): Entry[] {
	const out: Entry[] = [];
	schema.fields.forEach((f, i) => {
		const forms: [string, boolean][] = [
			[f.name, false],
			...(f.aliases ?? []).map((a): [string, boolean] => [a, true]),
		];
		for (const [surface, alias] of forms) {
			const words = wordsOf(surface);
			if (words.length > 0 && words.length <= MAX_WORDS)
				out.push({ words, field: i, kind: f.kind, alias, value: -1 });
		}
	});
	return out;
}

export function enumEntries(schema: Schema): Entry[] {
	const out: Entry[] = [];
	schema.fields.forEach((f, i) => {
		(f.values ?? []).forEach((v, j) => {
			const words = wordsOf(v);
			if (words.length > 0 && words.length <= MAX_WORDS)
				out.push({ words, field: i, kind: f.kind, alias: false, value: j });
		});
	});
	return out;
}

/** Every token except whitespace/newline runs: the sequence the model sees. */
export function modelTokens(text: string): Token[] {
	return tokenize(text).filter(
		(t) => t.cls !== CharClass.Space && t.cls !== CharClass.Newline,
	);
}

export function wordPositions(tokens: Token[]): number[] {
	const out: number[] = [];
	tokens.forEach((t, i) => {
		if (isWord(t)) out.push(i);
	});
	return out;
}

function connected(tokens: Token[], a: number, b: number): boolean {
	for (let k = a + 1; k < b; k++)
		if (!CONNECTORS.has(tokens[k]!.text)) return false;
	return true;
}

const same = (a: string[], b: string[]) =>
	a.length === b.length && a.every((w, i) => w === b[i]);

function matchAt(
	tokens: Token[],
	positions: number[],
	wi: number,
	entries: Entry[],
): Span | null {
	const lowered = tokens.map((t) => t.text.toLowerCase());
	for (let n = MAX_WORDS; n >= 1; n--) {
		if (wi + n > positions.length) continue;
		const idx = positions.slice(wi, wi + n);
		let ok = true;
		for (let k = 0; k < n - 1; k++)
			if (!connected(tokens, idx[k]!, idx[k + 1]!)) ok = false;
		if (!ok) continue;
		const spanWords = idx.map((i) => lowered[i]!);
		const stems = spanWords.map(stem);
		const start = idx[0]!;
		const end = idx[n - 1]! + 1;
		for (const quality of [EXACT, STEM]) {
			const hits = entries.filter(
				(e) =>
					e.words.length === n &&
					(quality === EXACT
						? same(e.words, spanWords)
						: same(e.words.map(stem), stems)),
			);
			if (hits.length > 0) {
				const first = hits[0]!;
				const owners = [...new Set(hits.map((e) => e.field))].sort(
					(a, b) => a - b,
				);
				return {
					start,
					end,
					field: first.field,
					kind: first.kind,
					quality,
					alias: first.alias,
					value: first.value,
					owners,
				};
			}
		}
		if (n === 1) {
			const w = spanWords[0]!;
			if (w.length >= MIN_PREFIX) {
				const hits = entries.filter(
					(e) =>
						e.words.length === 1 &&
						e.words[0]!.length >= MIN_PREFIX &&
						e.words[0]!.startsWith(w),
				);
				const keys = new Set(hits.map((e) => `${e.field}:${e.value}`));
				if (keys.size === 1) {
					const e = hits[0]!;
					return {
						start,
						end,
						field: e.field,
						kind: e.kind,
						quality: PREFIX,
						alias: e.alias,
						value: e.value,
						owners: [e.field],
					};
				}
			}
			if (w.length >= MIN_TYPO) {
				const hits = entries.filter(
					(e) => e.words.length === 1 && withinOneEdit(w, e.words[0]!),
				);
				const keys = new Set(hits.map((e) => `${e.field}:${e.value}`));
				if (keys.size === 1) {
					const e = hits[0]!;
					return {
						start,
						end,
						field: e.field,
						kind: e.kind,
						quality: TYPO,
						alias: e.alias,
						value: e.value,
						owners: [e.field],
					};
				}
			}
		}
	}
	return null;
}

/** Greedy left-to-right, longest-first matching. Spans never overlap. */
export function matchSpans(tokens: Token[], entries: Entry[]): Span[] {
	const positions = wordPositions(tokens);
	const spans: Span[] = [];
	let wi = 0;
	while (wi < positions.length) {
		const found = matchAt(tokens, positions, wi, entries);
		if (!found) {
			wi++;
			continue;
		}
		spans.push(found);
		while (wi < positions.length && positions[wi]! < found.end) wi++;
	}
	return spans;
}

/** Resolve an already-extracted phrase (compiler side). Indices are relative to the phrase. */
export function resolveWords(words: string[], entries: Entry[]): Span | null {
	const toks = modelTokens(words.join(" "));
	const positions = wordPositions(toks);
	if (positions.length === 0) return null;
	return matchAt(toks, positions, 0, entries);
}
