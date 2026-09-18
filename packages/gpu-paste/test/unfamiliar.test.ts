import { describe, expect, it } from "vitest";
import { parse } from "../src/index.ts";
import cases from "./unfamiliar.json" with { type: "json" };

interface Case {
  text: string;
  kind: string;
  spans: { kind: string; text: string }[];
}

/**
 * End-to-end scores on the hand-written "unfamiliar" set (rules + model). These numbers are
 * reported in MODEL_CARD.md; the assertions are floors, not targets, and the set is never
 * used for tuning.
 */
describe("gpu-paste unfamiliar set", async () => {
  const all = cases as Case[];
  let correct = 0;
  const confusion: Record<string, Record<string, number>> = {};
  const tp: Record<string, number> = {};
  const fp: Record<string, number> = {};
  const fn: Record<string, number> = {};
  const misses: string[] = [];
  for (const c of all) {
    const r = await parse(c.text, { backend: "cpu" });
    confusion[c.kind] ??= {};
    confusion[c.kind]![r.kind] = (confusion[c.kind]![r.kind] ?? 0) + 1;
    if (r.kind === c.kind) correct++;
    else misses.push(`${JSON.stringify(c.text.slice(0, 40))}: expected ${c.kind}, got ${r.kind}`);
    const gold = new Set(
      c.spans.map((s) => {
        const start = c.text.indexOf(s.text);
        return `${s.kind}:${start}:${start + s.text.length}`;
      }),
    );
    const pred = new Set(r.spans.map((s) => `${s.kind}:${s.span[0]}:${s.span[1]}`));
    for (const p of pred) {
      const kind = p.split(":")[0]!;
      if (gold.has(p)) tp[kind] = (tp[kind] ?? 0) + 1;
      else fp[kind] = (fp[kind] ?? 0) + 1;
    }
    for (const g of gold) if (!pred.has(g)) fn[g.split(":")[0]!] = (fn[g.split(":")[0]!] ?? 0) + 1;
  }
  const kinds = [...new Set([...Object.keys(tp), ...Object.keys(fp), ...Object.keys(fn)])].sort();
  const rows = kinds.map((k) => {
    const p = (tp[k] ?? 0) / Math.max(1, (tp[k] ?? 0) + (fp[k] ?? 0));
    const r = (tp[k] ?? 0) / Math.max(1, (tp[k] ?? 0) + (fn[k] ?? 0));
    const f1 = p + r ? (2 * p * r) / (p + r) : 0;
    return {
      kind: k,
      precision: p.toFixed(2),
      recall: r.toFixed(2),
      f1: f1.toFixed(2),
      support: (tp[k] ?? 0) + (fn[k] ?? 0),
    };
  });
  const sum = (o: Record<string, number>) => Object.values(o).reduce((a, b) => a + b, 0);
  const mp = sum(tp) / Math.max(1, sum(tp) + sum(fp));
  const mr = sum(tp) / Math.max(1, sum(tp) + sum(fn));
  const microF1 = mp + mr ? (2 * mp * mr) / (mp + mr) : 0;
  const accuracy = correct / all.length;
  console.log(
    `unfamiliar: ${all.length} cases, kind accuracy ${(accuracy * 100).toFixed(1)}%, span micro-F1 ${(microF1 * 100).toFixed(1)}%`,
  );
  console.table(rows);
  if (misses.length) console.log(`kind misses:\n  ${misses.join("\n  ")}`);

  it("has at least 60 cases", () => {
    expect(all.length).toBeGreaterThanOrEqual(60);
  });
  it("kind accuracy floor", () => {
    expect(accuracy).toBeGreaterThanOrEqual(0.7);
  });
  it("span micro-F1 floor", () => {
    expect(microF1).toBeGreaterThanOrEqual(0.5);
  });
});
