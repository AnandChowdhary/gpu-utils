import { describe, expect, it } from "vitest";
import { parseMany } from "../src/index.ts";
import type { Schema } from "../src/match.ts";
import { canon, read, strip } from "./helpers.ts";

type Gold = { text: string; schema: Schema; now: string; spec: unknown; domain?: string };
type Unfamiliar = {
  schemas: Record<string, Schema>;
  now: string;
  cases: { text: string; schema: string; spec: unknown }[];
};

async function exact(
  cases: Gold[],
  withSpans: boolean,
): Promise<{ rate: number; failures: string[] }> {
  let ok = 0;
  const failures: string[] = [];
  for (const c of cases) {
    const [spec] = await parseMany([c.text], { schema: c.schema, now: c.now, backend: "cpu" });
    if (canon(strip(spec!, withSpans)) === canon(c.spec)) ok++;
    else
      failures.push(
        `${c.text}\n   got  ${canon(strip(spec!, withSpans))}\n   want ${canon(c.spec)}`,
      );
  }
  return { rate: ok / cases.length, failures };
}

const report = (name: string, rate: number, n: number, failures: string[], show: number) =>
  process.stdout.write(
    `${name} exact match: ${(rate * 100).toFixed(1)}% (${n})\n${failures.slice(0, show).join("\n")}\n`,
  );

/** Floors are regression guards well below the numbers in MODEL_CARD.md, not targets. */
describe("gpu-view evaluation", () => {
  it("held-out schemas (transfer): spec exact match", async () => {
    const cases = read<Gold[]>("eval/heldout.json");
    const { rate, failures } = await exact(cases, true);
    report("transfer", rate, cases.length, failures, 8);
    expect(rate).toBeGreaterThan(0.6);
  });
  it("in-domain schemas: spec exact match", async () => {
    const cases = read<Gold[]>("eval/indomain.json");
    const { rate, failures } = await exact(cases, true);
    report("in-domain", rate, cases.length, failures, 5);
    expect(rate).toBeGreaterThan(0.6);
  });
  it("unfamiliar hand-written phrases: spec exact match (spans ignored)", async () => {
    const u = read<Unfamiliar>("eval/unfamiliar.json");
    expect(u.cases.length).toBeGreaterThanOrEqual(60);
    const cases: Gold[] = u.cases.map((c) => ({
      text: c.text,
      schema: u.schemas[c.schema]!,
      now: u.now,
      spec: c.spec,
    }));
    const { rate, failures } = await exact(cases, false);
    report("unfamiliar", rate, cases.length, failures, failures.length);
    expect(rate).toBeGreaterThan(0.4);
  });
});
