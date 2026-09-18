import { type FeatureRows, viterbi } from "@gpu-utils/runtime";
import type { Logits } from "./cpu.ts";
import { findArxiv, findDois, findUrls, type Span } from "./features.ts";
import type { Model } from "./model.ts";

export type CiteType =
	| "article"
	| "book"
	| "chapter"
	| "conference"
	| "thesis"
	| "report"
	| "web"
	| "preprint"
	| "unknown";

export interface Person {
	given?: string;
	family?: string;
	/** Set when the name could not be split (organisations, single tokens). */
	literal?: string;
	/** UTF-16 offsets of this name in the input, half-open. */
	span: Span;
}

export type CiteField =
	| "authors"
	| "editors"
	| "title"
	| "container"
	| "year"
	| "volume"
	| "issue"
	| "pages"
	| "publisher"
	| "location"
	| "edition"
	| "doi"
	| "arxiv"
	| "url"
	| "accessed";

export interface CiteDiagnostics {
	/** Mean per-token probability of the decoded tag sequence (0–1). */
	confidence: number;
	/** Softmax probability of the chosen document type (0–1). */
	typeConfidence: number;
	/** The author list was truncated with "et al." or similar. */
	etAl: boolean;
	warnings: string[];
	/** One BIO tag per token, aligned with `tokens`. */
	tags: string[];
	tokens: { text: string; start: number; end: number }[];
}

export interface CiteRecord {
	type: CiteType;
	authors: Person[];
	editors?: Person[];
	title?: string;
	/** Journal, book or proceedings title, or website name. */
	container?: string;
	year?: number;
	volume?: string;
	issue?: string;
	pages?: { from: string; to: string };
	publisher?: string;
	location?: string;
	edition?: string;
	doi?: string;
	arxiv?: string;
	url?: string;
	/** Access date as written, e.g. "March 3, 2021". */
	accessed?: string;
	/** UTF-16 offsets of every extracted field in the input, half-open. */
	spans: Partial<Record<CiteField, Span>>;
	/** Offsets of this reference within the text passed to parse/parseMany. */
	range: Span;
	diagnostics: CiteDiagnostics;
}

const ROLE_FIELD: Record<string, CiteField> = {
	AUTHOR: "authors",
	EDITOR: "editors",
	TITLE: "title",
	CONTAINER: "container",
	YEAR: "year",
	VOLUME: "volume",
	ISSUE: "issue",
	PAGES: "pages",
	PUBLISHER: "publisher",
	LOCATION: "location",
	EDITION: "edition",
	DOI: "doi",
	ARXIV: "arxiv",
	URL: "url",
	ACCESSED: "accessed",
};

const NEG = -1e4;
const transitionCache = new WeakMap<Model, Float32Array>();

/** Learned CRF transitions plus hard BIO constraints (O→I-X, B-X→I-Y, I-X→I-Y forbidden). */
function constrainedTransitions(model: Model): Float32Array {
	let cached = transitionCache.get(model);
	if (cached) return cached;
	const K = model.manifest.labels.length;
	const R = model.manifest.roles.length;
	cached = new Float32Array(model.t.trans!);
	for (let to = 1 + R; to < K; to++) {
		const role = to - 1 - R;
		cached[0 * K + to] = NEG;
		for (let from = 1; from < K; from++)
			if ((from - 1) % R !== role) cached[from * K + to] = NEG;
	}
	transitionCache.set(model, cached);
	return cached;
}

interface Entity {
	role: string;
	first: number; // token index
	last: number; // inclusive
}

function entities(path: Int32Array, model: Model): Entity[] {
	const R = model.manifest.roles.length;
	const out: Entity[] = [];
	let cur: Entity | undefined;
	for (let i = 0; i < path.length; i++) {
		const tag = path[i]!;
		if (tag === 0) {
			cur = undefined;
			continue;
		}
		const role = model.manifest.roles[(tag - 1) % R]!;
		const begin = tag <= R;
		if (begin || !cur || cur.role !== role) {
			cur = { role, first: i, last: i };
			out.push(cur);
		} else cur.last = i;
	}
	return out;
}

const LEAD = " \t\"'“”‘’«»‚„*_";
const TRAIL = " \t.,;:\"'“”‘’«»‚„*_";

/** True when the whole value is wrapped in exactly one pair of brackets, e.g. "(2019)". */
function wrapped(inner: string, open: string, close: string): boolean {
	return (
		inner.startsWith(open) &&
		inner.endsWith(close) &&
		inner.indexOf(close) === inner.length - 1 &&
		inner.lastIndexOf(open) === 0
	);
}

/** Trim quotes, whitespace and trailing punctuation; drop unbalanced or wrapping brackets. */
export function trimSpan(text: string, start: number, end: number): Span {
	let s = start;
	let e = end;
	for (let iter = 0; iter < 4; iter++) {
		while (s < e && LEAD.includes(text[s]!)) s++;
		while (e > s && TRAIL.includes(text[e - 1]!)) e--;
		const inner = text.slice(s, e);
		if (
			(inner.startsWith("(") && !inner.includes(")")) ||
			(inner.startsWith("[") && !inner.includes("]"))
		)
			s++;
		else if (
			(inner.endsWith(")") && !inner.includes("(")) ||
			(inner.endsWith("]") && !inner.includes("["))
		)
			e--;
		else if (wrapped(inner, "(", ")") || wrapped(inner, "[", "]")) {
			s++;
			e--;
		} else break;
	}
	return [s, e];
}

const clean = (s: string): string => s.replace(/\s+/g, " ").trim();
const YEAR_RE = /(?:1[5-9]|20)\d\d/;
const VOL_CUE = /^(?:vol\.?|volume|vols\.?|bd\.?|jg\.?|band|v\.)\s*/i;
const ISSUE_CUE = /^(?:no\.?|number|issue|nr\.?|num\.?|heft|h\.|#)\s*/i;
const PAGE_CUE = /^(?:pp?\.?|pages?|s\.|pg\.?)\s*/i;
const EDITION_CUE = /\s*(?:ed\.?|edn\.?|edition|aufl\.?|auflage)\.?$/i;
const ET_AL = /\bet\s+al\b|\band others\b|\bu\.\s?a\.|\bet al\./i;
const DOI_VALUE = /^10\.\d{4,9}\/\S+$/;
const ARXIV_VALUE =
	/^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Za-z]{2})?\/\d{7}(?:v\d+)?)$/i;
const URL_VALUE = /^(?:https?:\/\/|www\.)\S+$/i;

export function parsePages(value: string): { from: string; to: string } {
	const m = /^(.+?)\s*(?:[-–—‐‑]+|\bto\b)\s*(.+)$/.exec(value);
	if (!m) return { from: value, to: value };
	const from = m[1]!.trim();
	let to = m[2]!.trim();
	if (/^\d+$/.test(from) && /^\d+$/.test(to) && to.length < from.length) {
		to = from.slice(0, from.length - to.length) + to;
	}
	return { from, to };
}

function softmaxAt(
	v: Float32Array,
	offset: number,
	k: number,
	index: number,
): number {
	let max = -Infinity;
	for (let j = 0; j < k; j++) if (v[offset + j]! > max) max = v[offset + j]!;
	let sum = 0;
	for (let j = 0; j < k; j++) sum += Math.exp(v[offset + j]! - max);
	return Math.exp(v[offset + index]! - max) / sum;
}

function overlaps(a: Span, b: Span): boolean {
	return a[0] < b[1] && b[0] < a[1];
}

/** Turns model logits into the typed record. All validation and field semantics live here. */
export function decode(
	model: Model,
	features: FeatureRows,
	logits: Logits,
	text: string,
	base = 0,
): CiteRecord {
	const m = model.manifest;
	const K = m.labels.length;
	const R = m.roles.length;
	const P = m.nameparts.length;
	const n = features.tokens.length;
	const tokens = features.tokens;
	const warnings: string[] = [];
	const spans: Partial<Record<CiteField, Span>> = {};

	// 1. Viterbi over constrained CRF transitions; a sequence may not start inside an entity.
	const emissions = new Float32Array(logits.tags);
	for (let j = 1 + R; j < K; j++) emissions[j] = NEG;
	const path =
		n > 0
			? viterbi(emissions, n, K, constrainedTransitions(model))
			: new Int32Array(0);
	let confidence = 0;
	for (let i = 0; i < n; i++)
		confidence += softmaxAt(logits.tags, i * K, K, path[i]!);
	confidence = n > 0 ? confidence / n : 0;

	// 2. Group tags into entities and pick one per single-valued field.
	const ents = entities(path, model);
	const byRole = new Map<string, Entity[]>();
	for (const e of ents) {
		const list = byRole.get(e.role) ?? [];
		list.push(e);
		byRole.set(e.role, list);
	}
	const abs = (s: Span): Span => [s[0] + base, s[1] + base];
	const entSpan = (e: Entity): Span =>
		trimSpan(text, tokens[e.first]!.start, tokens[e.last]!.end);
	const pick = (role: string, longest: boolean): Entity | undefined => {
		const list = byRole.get(role);
		if (!list || list.length === 0) return undefined;
		if (!longest) return list[0];
		let best = list[0]!;
		for (const e of list)
			if (
				tokens[e.last]!.end - tokens[e.first]!.start >
				tokens[best.last]!.end - tokens[best.first]!.start
			)
				best = e;
		return best;
	};
	const fieldValue = (role: string, longest = false): string | undefined => {
		const e = pick(role, longest);
		if (!e) return undefined;
		const sp = entSpan(e);
		if (sp[0] >= sp[1]) return undefined;
		spans[ROLE_FIELD[role]!] = abs(sp);
		return clean(text.slice(sp[0], sp[1]));
	};

	// 3. People: entity boundaries from BIO tags, given/family split from the name-part head.
	const people = (role: string): Person[] => {
		const out: Person[] = [];
		for (const e of byRole.get(role) ?? []) {
			const sp = entSpan(e);
			if (sp[0] >= sp[1]) continue;
			let given = "";
			let family = "";
			for (let i = e.first; i <= e.last; i++) {
				let part = 0;
				let best = -Infinity;
				for (let j = 0; j < P; j++) {
					const v = logits.parts[i * P + j]!;
					if (v > best) {
						best = v;
						part = j;
					}
				}
				if (part === 1) given += tokens[i]!.text;
				else if (part === 2) family += tokens[i]!.text;
			}
			given = clean(given)
				.replace(/[,;]+$/, "")
				.trim();
			family = clean(family)
				.replace(/[,;]+$/, "")
				.trim();
			const person: Person = { span: abs(sp) };
			if (!given && !family) person.literal = clean(text.slice(sp[0], sp[1]));
			else {
				if (given) person.given = given;
				if (family) person.family = family;
			}
			out.push(person);
		}
		return out;
	};
	const authors = people("AUTHOR");
	const editors = people("EDITOR");
	if (authors.length > 0)
		spans.authors = [authors[0]!.span[0], authors[authors.length - 1]!.span[1]];
	if (editors.length > 0)
		spans.editors = [editors[0]!.span[0], editors[editors.length - 1]!.span[1]];

	// 4. Document type from the pooled head; "unknown" below 50% confidence.
	const T = m.types.length;
	let typeIndex = 0;
	for (let j = 1; j < T; j++)
		if (logits.type[j]! > logits.type[typeIndex]!) typeIndex = j;
	const typeConfidence = n > 0 ? softmaxAt(logits.type, 0, T, typeIndex) : 0;
	const type: CiteType =
		n > 0 && typeConfidence >= 0.5
			? (m.types[typeIndex] as CiteType)
			: "unknown";
	if (n > 0 && typeConfidence < 0.6) warnings.push("uncertain document type");

	const record: CiteRecord = {
		type,
		authors,
		spans,
		range: [base, base + text.length],
		diagnostics: {
			confidence,
			typeConfidence,
			etAl: ET_AL.test(text),
			warnings,
			tags: Array.from(path, (i) => m.labels[i] ?? "O"),
			tokens: tokens.map(({ text: t, start, end }) => ({
				text: t,
				start: start + base,
				end: end + base,
			})),
		},
	};
	if (editors.length > 0) record.editors = editors;

	// 5. Text fields.
	const title = fieldValue("TITLE", true);
	if (title) record.title = title;
	const container = fieldValue("CONTAINER", true);
	if (container) record.container = container;
	const publisher = fieldValue("PUBLISHER");
	if (publisher) record.publisher = publisher;
	const location = fieldValue("LOCATION");
	if (location) record.location = location;
	const accessed = fieldValue("ACCESSED");
	if (accessed) record.accessed = accessed;

	// 6. Numeric-ish fields with cue words stripped.
	const yearText = fieldValue("YEAR");
	if (yearText !== undefined) {
		const y = YEAR_RE.exec(yearText);
		if (y) record.year = Number(y[0]);
		else if (/n\.\s?d\b|no date|s\.\s?d\./i.test(yearText))
			warnings.push("no date (n.d.)");
		else warnings.push(`date not numeric: "${yearText}"`);
	} else warnings.push("no year found");
	const volume = fieldValue("VOLUME");
	if (volume) record.volume = volume.replace(VOL_CUE, "");
	const issue = fieldValue("ISSUE");
	if (issue) record.issue = issue.replace(ISSUE_CUE, "");
	const pages = fieldValue("PAGES");
	if (pages) record.pages = parsePages(pages.replace(PAGE_CUE, ""));
	const edition = fieldValue("EDITION");
	if (edition) record.edition = edition.replace(EDITION_CUE, "");

	// 7. Identifiers: deterministic regexes first, the model's span as a validated fallback.
	const doiEnt = pick("DOI", false);
	const dois = findDois(text);
	let doi = doiEnt ? dois.find((d) => overlaps(d, entSpan(doiEnt))) : undefined;
	doi ??= dois[0];
	if (!doi && doiEnt) {
		const sp = entSpan(doiEnt);
		if (DOI_VALUE.test(text.slice(sp[0], sp[1]))) doi = sp;
	}
	if (doi) {
		record.doi = text.slice(doi[0], doi[1]);
		spans.doi = abs(doi);
	}
	const arxEnt = pick("ARXIV", false);
	const arxs = findArxiv(text);
	let arx = arxEnt ? arxs.find((a) => overlaps(a, entSpan(arxEnt))) : undefined;
	arx ??= arxs[0];
	if (!arx && arxEnt) {
		const sp = entSpan(arxEnt);
		if (ARXIV_VALUE.test(text.slice(sp[0], sp[1]))) arx = sp;
	}
	if (arx) {
		record.arxiv = text.slice(arx[0], arx[1]);
		spans.arxiv = abs(arx);
	}
	const urlEnt = pick("URL", false);
	const urls = findUrls(text).filter((u) => !(doi && overlaps(u, doi)));
	let url = urlEnt ? urls.find((u) => overlaps(u, entSpan(urlEnt))) : undefined;
	url ??= urls[0];
	if (!url && urlEnt) {
		const sp = entSpan(urlEnt);
		if (URL_VALUE.test(text.slice(sp[0], sp[1]))) url = sp;
	}
	if (url) {
		record.url = text.slice(url[0], url[1]);
		spans.url = abs(url);
	}

	if (!record.title && !record.url) warnings.push("no title found");
	if (authors.length === 0 && type !== "web" && type !== "unknown")
		warnings.push("no authors found");
	if (record.diagnostics.etAl)
		warnings.push("author list truncated with et al.");
	if (n === 0) warnings.push("empty input");
	return record;
}
