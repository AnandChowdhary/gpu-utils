/**
 * gpu-paste: Understand pasted text: detect what it is and extract structured fields, on-device.
 *
 * Layering (see README "Rules vs. learned"):
 *   1. rules.ts decides every kind that can be validated deterministically (JSON, CSV/TSV,
 *      HTML, URL, email, phone, datetime, color, UUID, JWT, IP, path, money, number, empty)
 *      and finds regex spans (email, url, uuid, ip, color, ISO date, hashtag, mention,
 *      issue_ref, commit).
 *   2. The model classifies the ambiguous kinds (address, contact, prose, list, code,
 *      markdown) and tags person / company / address / date / money / phone spans.
 *   3. decode.ts + this file compile spans into `parsed`.
 */
import { type Backend, hasWebGPU, type Token } from "@gpu-utils/runtime";
import { forwardCpu, type Logits } from "./cpu.ts";
import { decodeSpans, type ModelSpan, mergeSpans, softmax } from "./decode.ts";
import { featurize } from "./features.ts";
import { forwardGpu } from "./gpu.ts";
import { MODEL } from "./model.ts";
import { detectWhole, type RuleSpan } from "./rules.ts";

export type { RuleSpan as Span } from "./rules.ts";
export {
	parseAmount,
	parseColor,
	parseDate,
	parseMoney,
	parseNumber,
	parsePhone,
} from "./rules.ts";

export type PasteKind =
	| "json"
	| "csv"
	| "tsv"
	| "markdown"
	| "html"
	| "code"
	| "url"
	| "email"
	| "phone"
	| "address"
	| "contact"
	| "datetime"
	| "color"
	| "uuid"
	| "jwt"
	| "ip"
	| "path"
	| "money"
	| "number"
	| "list"
	| "prose"
	| "empty";

export interface PasteDiagnostics {
	/** "rule:<detector>" when a deterministic detector decided the kind, else "model". */
	decidedBy: string;
	backend: "cpu" | "webgpu" | "none";
	tokens: number;
	/** Softmax over the learned kinds, when the model ran. */
	kindProbabilities?: Record<string, number>;
	/** Number of 512-token windows the model ran over. */
	windows?: number;
	notes: string[];
	ms: number;
}

export interface PasteResult {
	kind: PasteKind;
	confidence: number;
	spans: RuleSpan[];
	parsed?: unknown;
	diagnostics: PasteDiagnostics;
}

export interface PasteOptions {
	/** "auto" uses WebGPU for large inputs when available and the CPU reference otherwise. */
	backend?: Backend;
}

/** Inputs shorter than this run on the CPU under "auto": GPU readback latency dominates. */
const GPU_MIN_TOKENS = 256;
/** The model runs over windows of this many tokens; kind logits are averaged across windows. */
const WINDOW = 512;

const PROMOTE: Record<string, PasteKind> = {
	date: "datetime",
	money: "money",
	phone: "phone",
	address: "address",
	person: "contact",
	company: "contact",
};

async function runModel(
	tokens: Token[],
	rows: number[][],
	useGpu: boolean,
): Promise<{ spans: ModelSpan[]; kind: Float32Array; windows: number }> {
	const K = MODEL.manifest.kinds.length;
	const kind = new Float32Array(K);
	const spans: ModelSpan[] = [];
	let windows = 0;
	for (let start = 0; start < rows.length; start += WINDOW) {
		const rowSlice = rows.slice(start, start + WINDOW);
		const tokenSlice = tokens.slice(start, start + WINDOW);
		const logits: Logits = useGpu
			? await forwardGpu(MODEL, rowSlice)
			: forwardCpu(MODEL, rowSlice);
		for (let k = 0; k < K; k++) kind[k] = kind[k]! + logits.kind[k]!;
		spans.push(...decodeSpans(MODEL, tokenSlice, logits.span));
		windows++;
	}
	if (windows > 1) for (let k = 0; k < K; k++) kind[k] = kind[k]! / windows;
	return { spans, kind, windows };
}

function buildParsed(
	kind: PasteKind,
	text: string,
	spans: RuleSpan[],
): unknown {
	const pick = (k: string) =>
		spans
			.filter((s) => s.kind === k)
			.map((s) => s.value ?? text.slice(s.span[0], s.span[1]));
	if (kind === "contact") {
		const out: Record<string, string | undefined> = {};
		const fields: [string, string][] = [
			["name", "person"],
			["company", "company"],
			["email", "email"],
			["phone", "phone"],
			["url", "url"],
			["address", "address"],
		];
		for (const [field, spanKind] of fields) {
			const v = pick(spanKind)[0];
			if (v !== undefined) out[field] = v;
		}
		return out;
	}
	if (kind === "address") {
		const a = pick("address")[0] ?? text.trim();
		return {
			lines: a
				.split(/\n|,\s*/)
				.map((l) => l.trim())
				.filter(Boolean),
		};
	}
	if (kind === "list") {
		const lines = text
			.split(/\r?\n/)
			.map((l) =>
				l
					.replace(
						/^\s*(?:[-*•·–—+>]+|\d+[.)]|[a-z][.)]|\[[ x]\]|[☐✅✓→]|\(\d+\))\s*/i,
						"",
					)
					.trim(),
			)
			.filter(Boolean);
		const items =
			lines.length > 1
				? lines
				: text
						.split(/\s*[,;|•]\s*|\s+\/\s+/)
						.map((s) => s.trim())
						.filter(Boolean);
		return { items };
	}
	if (kind === "markdown") {
		return {
			headings: [...text.matchAll(/^#{1,6}\s+(.+)$/gm)].map((m) =>
				m[1]!.trim(),
			),
			links: [...text.matchAll(/\[([^\]]*)\]\(([^)\s]+)\)/g)].map((m) => ({
				text: m[1]!,
				href: m[2]!,
			})),
		};
	}
	if (kind === "code") {
		return { lines: text.split(/\r?\n/).length };
	}
	return undefined;
}

export async function parse(
	text: string,
	options: PasteOptions = {},
): Promise<PasteResult> {
	const t0 = performance.now();
	const notes: string[] = [];
	const diag = (
		decidedBy: string,
		backend: "cpu" | "webgpu" | "none",
		tokens: number,
	): PasteDiagnostics => ({
		decidedBy,
		backend,
		tokens,
		notes,
		ms: performance.now() - t0,
	});
	if (!text.trim()) {
		return {
			kind: "empty",
			confidence: 1,
			spans: [],
			diagnostics: diag("rule:empty", "none", 0),
		};
	}
	const rule = detectWhole(text);
	if (rule?.note) notes.push(rule.note);
	const trimmedStart = text.length - text.trimStart().length;
	const whole: [number, number] = [
		trimmedStart,
		trimmedStart + text.trim().length,
	];
	// Single-entity kinds: the whole paste is the span; no model needed.
	if (rule?.spanKind) {
		const span: RuleSpan = { kind: rule.spanKind, span: whole };
		if (rule.value !== undefined) span.value = rule.value;
		return {
			kind: rule.kind,
			confidence: rule.confidence,
			spans: [span],
			parsed: rule.parsed,
			diagnostics: diag(`rule:${rule.kind}`, "none", 0),
		};
	}
	if (
		rule &&
		(rule.kind === "number" || rule.kind === "path" || rule.kind === "jwt")
	) {
		return {
			kind: rule.kind,
			confidence: rule.confidence,
			spans: [],
			parsed: rule.parsed,
			diagnostics: diag(`rule:${rule.kind}`, "none", 0),
		};
	}
	// Everything else gets span-tagged by rules + model; structured kinds keep the rule's kind.
	const { tokens, rows } = featurize(text);
	const backend = options.backend ?? "auto";
	const useGpu =
		backend === "webgpu" ||
		(backend === "auto" && hasWebGPU() && tokens.length >= GPU_MIN_TOKENS);
	const model = await runModel(tokens, rows, useGpu);
	const spans = mergeSpans(text, model.spans);
	const probs = softmax(model.kind);
	const kindProbabilities = Object.fromEntries(
		MODEL.manifest.kinds.map((k, i) => [k, probs[i]!]),
	);
	const diagnostics = diag(
		rule ? `rule:${rule.kind}` : "model",
		useGpu ? "webgpu" : "cpu",
		tokens.length,
	);
	diagnostics.kindProbabilities = kindProbabilities;
	diagnostics.windows = model.windows;
	if (rule) {
		return {
			kind: rule.kind,
			confidence: rule.confidence,
			spans,
			parsed: rule.parsed,
			diagnostics,
		};
	}
	let best = 0;
	for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
	let kind = MODEL.manifest.kinds[best] as PasteKind;
	let confidence = probs[best]!;
	// Single-entity promotion: one learned span covering (nearly) the whole paste decides the kind.
	const trimmed = text.trim();
	if (
		spans.length === 1 &&
		(kind === "prose" || kind === "contact" || kind === "address")
	) {
		const s = spans[0]!;
		const covered =
			text.slice(s.span[0], s.span[1]).trim().length / trimmed.length;
		const promoted = PROMOTE[s.kind];
		if (promoted && covered >= 0.9 && promoted !== kind) {
			notes.push(
				`promoted ${kind} → ${promoted}: the whole paste is one ${s.kind} span`,
			);
			kind = promoted;
			confidence = Math.max(confidence, 0.6);
			diagnostics.decidedBy = "model+promotion";
		}
	}
	const parsed =
		kind === "datetime" || kind === "money" || kind === "phone"
			? spans[0]?.value
			: buildParsed(kind, text, spans);
	const result: PasteResult = { kind, confidence, spans, diagnostics };
	if (parsed !== undefined) result.parsed = parsed;
	return result;
}
