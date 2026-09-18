/**
 * Relative date resolver. Mirrors training/gpu_view/timeres.py exactly (fixtures in
 * test/fixtures/time.json). Input is the text of a TIME_VALUE span; output is an
 * inclusive [from, to] pair of ISO dates. Weeks start on Monday; all arithmetic is on
 * UTC calendar dates.
 */
import { MONTH_ABBREV, MONTHS } from "./lexicon.ts";

export type DateRange = [string, string];

const UNITS: Record<string, string> = {
	day: "day",
	days: "day",
	week: "week",
	weeks: "week",
	month: "month",
	months: "month",
	year: "year",
	years: "year",
	quarter: "quarter",
	quarters: "quarter",
};
const STRIP_LEADING = [
	"the ",
	"in ",
	"on ",
	"during ",
	"of ",
	"within ",
	"for ",
	"from ",
	"since ",
];

const utc = (y: number, m: number, d: number) =>
	new Date(Date.UTC(y, m - 1, d));
const iso = (d: Date) => d.toISOString().slice(0, 10);
const addDays = (d: Date, n: number) => new Date(d.getTime() + n * 86400000);
const daysInMonth = (y: number, m: number) =>
	new Date(Date.UTC(y, m, 0)).getUTCDate();

function addMonths(d: Date, n: number): Date {
	const m0 = d.getUTCMonth() + n;
	const y = d.getUTCFullYear() + Math.floor(m0 / 12);
	const m = (((m0 % 12) + 12) % 12) + 1;
	return utc(y, m, Math.min(d.getUTCDate(), daysInMonth(y, m)));
}
const monthRange = (y: number, m: number): [Date, Date] => [
	utc(y, m, 1),
	utc(y, m, daysInMonth(y, m)),
];
const quarterRange = (y: number, q: number): [Date, Date] => {
	const m = (q - 1) * 3 + 1;
	return [utc(y, m, 1), monthRange(y, m + 2)[1]];
};
function weekRange(d: Date): [Date, Date] {
	const start = addDays(d, -((d.getUTCDay() + 6) % 7));
	return [start, addDays(start, 6)];
}
const yearRange = (y: number): [Date, Date] => [utc(y, 1, 1), utc(y, 12, 31)];

function period(unit: string, today: Date, shift: number): [Date, Date] {
	if (unit === "day") {
		const d = addDays(today, shift);
		return [d, d];
	}
	if (unit === "week") return weekRange(addDays(today, 7 * shift));
	if (unit === "month") {
		const d = addMonths(
			utc(today.getUTCFullYear(), today.getUTCMonth() + 1, 1),
			shift,
		);
		return monthRange(d.getUTCFullYear(), d.getUTCMonth() + 1);
	}
	if (unit === "quarter") {
		const q = Math.floor(today.getUTCMonth() / 3) + shift;
		const y = today.getUTCFullYear() + Math.floor(q / 4);
		return quarterRange(y, (((q % 4) + 4) % 4) + 1);
	}
	return yearRange(today.getUTCFullYear() + shift);
}

function monthIndex(word: string): number {
	const i = MONTHS.indexOf(word);
	if (i >= 0) return i + 1;
	if (word === "sept") return 9;
	const j = MONTH_ABBREV.indexOf(word);
	return j >= 0 ? j + 1 : 0;
}

export function normalizeTime(text: string): string {
	let t = text.toLowerCase().replace(/\s+/g, " ").trim();
	t = t.replace(/^[ ,.!?;:"'()]+|[ ,.!?;:"'()]+$/g, "");
	let changed = true;
	while (changed) {
		changed = false;
		for (const lead of STRIP_LEADING) {
			if (t.startsWith(lead)) {
				t = t.slice(lead.length);
				changed = true;
			}
		}
	}
	return t.trim();
}

/** `today` is any Date; only its UTC calendar date is used. */
export function resolveTime(text: string, today: Date): DateRange | null {
	const t = normalizeTime(text);
	if (!t) return null;
	const day = utc(
		today.getUTCFullYear(),
		today.getUTCMonth() + 1,
		today.getUTCDate(),
	);
	const r = resolveNormalized(t, day);
	return r ? [iso(r[0]), iso(r[1])] : null;
}

function resolveNormalized(t: string, today: Date): [Date, Date] | null {
	const y = today.getUTCFullYear();
	if (t === "today" || t === "now") return [today, today];
	if (t === "yesterday") {
		const d = addDays(today, -1);
		return [d, d];
	}
	if (t === "tomorrow") {
		const d = addDays(today, 1);
		return [d, d];
	}
	if (t === "ytd" || t === "year to date") return [utc(y, 1, 1), today];
	if (t === "mtd" || t === "month to date")
		return [utc(y, today.getUTCMonth() + 1, 1), today];
	if (t === "qtd" || t === "quarter to date")
		return [period("quarter", today, 0)[0], today];

	let m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(t);
	if (m) {
		const [yy, mm, dd] = [Number(m[1]), Number(m[2]), Number(m[3])];
		if (mm < 1 || mm > 12 || dd < 1 || dd > daysInMonth(yy, mm)) return null;
		const d = utc(yy, mm, dd);
		return [d, d];
	}
	m = /^(\d{4})-(\d{2})$/.exec(t);
	if (m && Number(m[2]) >= 1 && Number(m[2]) <= 12)
		return monthRange(Number(m[1]), Number(m[2]));
	if (/^(19|20)\d{2}$/.test(t)) return yearRange(Number(t));

	m = /^(last|past|previous|prior|trailing) (\d+) (\w+)$/.exec(t);
	if (m && UNITS[m[3]!]) {
		const n = Number(m[2]);
		const unit = UNITS[m[3]!];
		if (n < 1) return null;
		if (unit === "day") return [addDays(today, -(n - 1)), today];
		if (unit === "week") return [addDays(today, -(7 * n - 1)), today];
		if (unit === "month") return [addDays(addMonths(today, -n), 1), today];
		if (unit === "quarter")
			return [addDays(addMonths(today, -3 * n), 1), today];
		return [addDays(addMonths(today, -12 * n), 1), today];
	}
	m = /^(next|coming|upcoming) (\d+) (\w+)$/.exec(t);
	if (m && UNITS[m[3]!]) {
		const n = Number(m[2]);
		const unit = UNITS[m[3]!];
		if (n < 1) return null;
		const start = addDays(today, 1);
		if (unit === "day") return [start, addDays(today, n)];
		if (unit === "week") return [start, addDays(today, 7 * n)];
		if (unit === "month") return [start, addMonths(today, n)];
		if (unit === "quarter") return [start, addMonths(today, 3 * n)];
		return [start, addMonths(today, 12 * n)];
	}
	m = /^(\d+) (\w+) ago$/.exec(t);
	if (m && UNITS[m[2]!]) {
		const n = Number(m[1]);
		const unit = UNITS[m[2]!];
		let d: Date;
		if (unit === "day") d = addDays(today, -n);
		else if (unit === "week") d = addDays(today, -7 * n);
		else if (unit === "month") d = addMonths(today, -n);
		else if (unit === "quarter") d = addMonths(today, -3 * n);
		else d = addMonths(today, -12 * n);
		return [d, d];
	}
	m =
		/^(this|current|last|past|previous|prior|next|coming|upcoming) (\w+)$/.exec(
			t,
		);
	if (m && UNITS[m[2]!]) {
		const unit = UNITS[m[2]!]!;
		const w = m[1]!;
		const shift =
			w === "this" || w === "current"
				? 0
				: ["last", "past", "previous", "prior"].includes(w)
					? -1
					: 1;
		return period(unit, today, shift);
	}
	m = /^q([1-4])(?: (?:of )?(\d{4}))?$/.exec(t);
	if (m) return quarterRange(m[2] ? Number(m[2]) : y, Number(m[1]));
	m = /^h([12])(?: (?:of )?(\d{4}))?$/.exec(t);
	if (m) {
		const yy = m[2] ? Number(m[2]) : y;
		return m[1] === "1"
			? [utc(yy, 1, 1), utc(yy, 6, 30)]
			: [utc(yy, 7, 1), utc(yy, 12, 31)];
	}
	m = /^([a-z]+)(?: (\d{4}))?$/.exec(t);
	if (m) {
		const mi = monthIndex(m[1]!);
		if (mi) return monthRange(m[2] ? Number(m[2]) : y, mi);
	}
	return null;
}
