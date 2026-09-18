import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { parse } from "../src/index.ts";

/**
 * Class-level evaluation with the promoted model: exact-set match and per-class
 * precision/recall on (a) the held-out generated set, (b) the frozen v1 unfamiliar set
 * (`eval/unfamiliar-v1.json`, treated as contaminated: its misses informed the v2
 * generator) and (c) the v2 unfamiliar set written before any v2 evaluation. Numbers go to
 * training/runs/eval.json and misses to training/runs/misses-<set>.txt for MODEL_CARD.md.
 * The held-out half is skipped when the generated cache is absent (CI).
 */
interface Case {
  text: string;
  classes: string[];
}
const root = resolve(import.meta.dirname, "..");
const runs = resolve(root, "training/runs");
const heldoutPath = resolve(root, "training/data/cache/heldout.jsonl");
const load = (name: string) =>
  JSON.parse(readFileSync(resolve(root, "eval", `${name}.json`), "utf8")) as Case[];

async function score(cases: Case[]) {
  let exact = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  const perClass = new Map<string, { tp: number; fp: number; fn: number }>();
  const bump = (c: string, k: "tp" | "fp" | "fn") => {
    const e = perClass.get(c) ?? { tp: 0, fp: 0, fn: 0 };
    e[k]++;
    perClass.set(c, e);
  };
  const misses: string[] = [];
  for (const c of cases) {
    const got = new Set((await parse(c.text, { backend: "cpu" })).classes);
    const want = new Set(c.classes);
    let ok = got.size === want.size;
    for (const g of got) {
      if (want.has(g)) {
        tp++;
        bump(g, "tp");
      } else {
        fp++;
        bump(g, "fp");
        ok = false;
      }
    }
    for (const w of want) {
      if (!got.has(w)) {
        fn++;
        bump(w, "fn");
        ok = false;
      }
    }
    if (ok) exact++;
    else misses.push(`${c.text}\n   want ${[...want].join(" ")}\n   got  ${[...got].join(" ")}`);
  }
  const precision = tp / Math.max(1, tp + fp);
  const recall = tp / Math.max(1, tp + fn);
  const f1 = (2 * precision * recall) / Math.max(1e-9, precision + recall);
  const macro = [...perClass.values()].map((e) => ({
    p: e.tp / Math.max(1, e.tp + e.fp),
    r: e.tp / Math.max(1, e.tp + e.fn),
  }));
  const macroP = macro.reduce((s, e) => s + e.p, 0) / Math.max(1, macro.length);
  const macroR = macro.reduce((s, e) => s + e.r, 0) / Math.max(1, macro.length);
  return {
    n: cases.length,
    exact: exact / cases.length,
    precision,
    recall,
    f1,
    macroPrecision: macroP,
    macroRecall: macroR,
    distinctClasses: perClass.size,
    misses,
  };
}

function record(key: string, r: Awaited<ReturnType<typeof score>>) {
  mkdirSync(runs, { recursive: true });
  const out = resolve(runs, "eval.json");
  const prev = existsSync(out) ? JSON.parse(readFileSync(out, "utf8")) : {};
  writeFileSync(
    out,
    `${JSON.stringify({ ...prev, [key]: { ...r, misses: undefined } }, null, 2)}\n`,
  );
  writeFileSync(resolve(runs, `misses-${key}.txt`), `${r.misses.join("\n")}\n`);
  console.log(
    `${key}: exact ${r.exact.toFixed(3)} P ${r.precision.toFixed(3)} R ${r.recall.toFixed(3)} F1 ${r.f1.toFixed(3)} (n=${r.n})`,
  );
}

describe("evaluation", () => {
  it("CPU latency on a 15-token phrase", async () => {
    const phrase = "bold red text, small, uppercase, blue on hover";
    await parse(phrase, { backend: "cpu" });
    const n = 300;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) await parse(phrase, { backend: "cpu" });
    const ms = (performance.now() - t0) / n;
    console.log(`cpu latency: ${ms.toFixed(3)} ms per parse (${n} runs)`);
    mkdirSync(runs, { recursive: true });
    const out = resolve(runs, "eval.json");
    const prev = existsSync(out) ? JSON.parse(readFileSync(out, "utf8")) : {};
    writeFileSync(out, `${JSON.stringify({ ...prev, cpuLatencyMs: ms }, null, 2)}\n`);
    expect(ms).toBeLessThan(50);
  });

  it("unfamiliar v1 (frozen, contaminated)", async () => {
    const cases = load("unfamiliar-v1");
    expect(cases.length).toBeGreaterThanOrEqual(60);
    const r = await score(cases);
    record("unfamiliarV1", r);
    expect(r.exact).toBeGreaterThan(0.3);
  });

  it("unfamiliar v2 (hand-written before v2 evaluation)", async () => {
    const cases = load("unfamiliar-v2");
    expect(cases.length).toBeGreaterThanOrEqual(66);
    const r = await score(cases);
    record("unfamiliarV2", r);
    expect(r.exact).toBeGreaterThan(0.2);
  });

  it.skipIf(!existsSync(heldoutPath))("held-out generated set", async () => {
    const rows = readFileSync(heldoutPath, "utf8")
      .trim()
      .split("\n")
      .slice(0, 3000)
      .map((l) => JSON.parse(l) as Case);
    const r = await score(rows);
    record("heldout", r);
    expect(r.exact).toBeGreaterThan(0.5);
  });
});
