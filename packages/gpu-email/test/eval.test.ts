import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { parse } from "../src/index.ts";
import { LINE_KINDS } from "../src/keywords.ts";

/**
 * Runs the shipped TypeScript decoder over the hand-written unfamiliar set (always) and
 * the generated held-out set (when `python -m gpu_email.data` has produced it) and
 * prints the same metrics as training/gpu_email/evaluate.py so the two decoders can be
 * cross-checked. Set GPU_EMAIL_EVAL=1 to include the (slow) held-out set.
 */
interface UnfamiliarCase {
  name: string;
  lines: [string, string][];
  contact?: Record<string, string | string[]> | null;
  newline?: string;
  trailing_newline?: boolean;
}
interface HeldoutCase {
  text: string;
  line_kinds: number[];
  reply: string;
  contact: Record<string, string | string[]>;
}

const DATA = resolve(import.meta.dirname, "../training/data");

function replyFromLines(lines: [string, string][]): string {
  const out: string[] = [];
  for (const [kind, text] of lines) {
    if (text.replace(/[ \t]/g, "") === "") {
      if (out.length && out[out.length - 1] !== "") out.push("");
    } else if (kind === "reply" || kind === "greeting" || kind === "closing") {
      out.push(text.trimEnd());
    } else if (out.length && out[out.length - 1] !== "") out.push("");
  }
  return out.join("\n").trim();
}

class Metrics {
  lineOk = 0;
  lineTotal = 0;
  replyOk = 0;
  replyTotal = 0;
  fields = new Map<string, { tp: number; fp: number; fn: number }>();
  failures: string[] = [];

  field(key: string) {
    let f = this.fields.get(key);
    if (!f) {
      f = { tp: 0, fp: 0, fn: 0 };
      this.fields.set(key, f);
    }
    return f;
  }

  addContact(
    gold: Record<string, string | string[]>,
    pred: Record<string, unknown> | undefined,
    name: string,
  ) {
    for (const key of ["name", "title", "company", "phone", "email", "url", "address"]) {
      const g = gold[key];
      const p = pred?.[key];
      const f = this.field(key);
      if (key === "phone" || key === "email" || key === "url") {
        const gs = new Set((g as string[] | undefined) ?? []);
        const ps = new Set((p as string[] | undefined) ?? []);
        for (const v of ps) gs.has(v) ? f.tp++ : f.fp++;
        for (const v of gs) if (!ps.has(v)) f.fn++;
        if ([...ps].some((v) => !gs.has(v)) || [...gs].some((v) => !ps.has(v)))
          this.failures.push(`[${key}] ${name}: expected ${[...gs]} got ${[...ps]}`);
      } else if (g && p) {
        if ((g as string).trim() === (p as string).trim()) f.tp++;
        else {
          f.fp++;
          f.fn++;
          this.failures.push(
            `[${key}] ${name}: expected ${JSON.stringify(g)} got ${JSON.stringify(p)}`,
          );
        }
      } else if (p) {
        f.fp++;
        this.failures.push(`[${key}] ${name}: expected none got ${JSON.stringify(p)}`);
      } else if (g) {
        f.fn++;
        this.failures.push(`[${key}] ${name}: expected ${JSON.stringify(g)} got none`);
      }
    }
  }

  report(title: string) {
    const rows = [`## ${title}`];
    rows.push(
      `line-kind accuracy: ${(this.lineOk / Math.max(1, this.lineTotal)).toFixed(4)} (${this.lineOk}/${this.lineTotal})`,
    );
    rows.push(
      `reply exact match: ${(this.replyOk / Math.max(1, this.replyTotal)).toFixed(4)} (${this.replyOk}/${this.replyTotal})`,
    );
    for (const [key, f] of this.fields) {
      if (f.tp + f.fp + f.fn === 0) continue;
      const p = f.tp / Math.max(1, f.tp + f.fp);
      const r = f.tp / Math.max(1, f.tp + f.fn);
      const f1 = (2 * p * r) / Math.max(1e-9, p + r);
      rows.push(
        `contact ${key} F1: ${f1.toFixed(3)} (P ${p.toFixed(3)} R ${r.toFixed(3)}, n=${f.tp + f.fn})`,
      );
    }
    console.log(`${rows.join("\n")}\n${this.failures.slice(0, 20).join("\n")}`);
  }
}

/** Reconstructs the per-line kind from segments (blank lines get -1). */
function lineKinds(text: string, segments: { kind: string; span: [number, number] }[]): number[] {
  const lines = text.split(/\r\n|\n/);
  const out: number[] = [];
  let offset = 0;
  for (const line of lines) {
    const nlLen = text.startsWith("\r\n", offset + line.length) ? 2 : 1;
    const seg = segments.find((s) => offset >= s.span[0] && offset < s.span[1]);
    out.push(
      line.replace(/[ \t]/g, "") === "" ? -1 : seg ? LINE_KINDS.indexOf(seg.kind as never) : -1,
    );
    offset += line.length + nlLen;
  }
  return out;
}

describe("gpu-email evaluation", () => {
  it("scores the unfamiliar set", async () => {
    const cases = JSON.parse(
      readFileSync(resolve(DATA, "unfamiliar.json"), "utf8"),
    ) as UnfamiliarCase[];
    expect(cases.length).toBeGreaterThanOrEqual(60);
    const m = new Metrics();
    for (const c of cases) {
      const nl = c.newline ?? "\n";
      const text = c.lines.map((l) => l[1]).join(nl) + (c.trailing_newline === false ? "" : nl);
      const r = await parse(text, { backend: "cpu" });
      const pred = lineKinds(text, r.segments);
      c.lines.forEach(([kind, t], i) => {
        if (t.replace(/[ \t]/g, "") === "") return;
        m.lineTotal++;
        if (pred[i] === LINE_KINDS.indexOf(kind as never)) m.lineOk++;
        else
          m.failures.push(
            `[line] ${c.name}#${i}: expected ${kind} got ${LINE_KINDS[pred[i]!] ?? "blank"}: ${JSON.stringify(t.slice(0, 40))}`,
          );
      });
      m.replyTotal++;
      const gold = replyFromLines(c.lines);
      if (r.reply === gold) m.replyOk++;
      else
        m.failures.push(
          `[reply] ${c.name}: expected ${JSON.stringify(gold.slice(0, 50))} got ${JSON.stringify(r.reply.slice(0, 50))}`,
        );
      m.addContact(c.contact ?? {}, r.contact as Record<string, unknown> | undefined, c.name);
    }
    m.report(`unfamiliar (${cases.length} emails, TypeScript decoder)`);
    expect(m.lineOk / m.lineTotal).toBeGreaterThan(0.5);
  });

  it.skipIf(!process.env.GPU_EMAIL_EVAL || !existsSync(resolve(DATA, "cache/heldout.json")))(
    "scores the generated held-out set",
    async () => {
      const cases = JSON.parse(
        readFileSync(resolve(DATA, "cache/heldout.json"), "utf8"),
      ) as HeldoutCase[];
      const m = new Metrics();
      for (const c of cases.slice(0, Number(process.env.GPU_EMAIL_EVAL_N ?? 1000))) {
        const r = await parse(c.text, { backend: "cpu" });
        const pred = lineKinds(c.text, r.segments);
        c.line_kinds.forEach((g, i) => {
          if (g < 0) return;
          m.lineTotal++;
          if (pred[i] === g) m.lineOk++;
        });
        m.replyTotal++;
        if (r.reply === c.reply) m.replyOk++;
        m.addContact(
          c.contact,
          r.contact as Record<string, unknown> | undefined,
          String(m.replyTotal),
        );
      }
      m.report(`held-out (${m.replyTotal} emails, TypeScript decoder)`);
      expect(m.lineOk / m.lineTotal).toBeGreaterThan(0.9);
    },
    600_000,
  );
});
