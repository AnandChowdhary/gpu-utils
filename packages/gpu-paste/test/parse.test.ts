import { describe, expect, it } from "vitest";
import { featurize } from "../src/features.ts";
import { parse } from "../src/index.ts";
import { parseAmount, parseDate, parseDelimited, parseMoney, ruleSpans } from "../src/rules.ts";

describe("gpu-paste featurizer", () => {
  it("emits one 10-id row per token", () => {
    const f = featurize("hello world\nx");
    expect(f.rows).toHaveLength(f.tokens.length);
    for (const r of f.rows) expect(r).toHaveLength(10);
  });
});

describe("gpu-paste rules", () => {
  const cases: [string, string][] = [
    ["", "empty"],
    ["   ", "empty"],
    ['{"a": [1, 2, {"b": null}]}', "json"],
    ["[1,2,3]", "json"],
    ["a,b,c\n1,2,3\n4,5,6", "csv"],
    ["a\tb\n1\t2", "tsv"],
    ["<p>hello <b>world</b></p>", "html"],
    ["https://example.com/path?q=1#frag", "url"],
    ["example.com/docs", "url"],
    ["someone@example.com", "email"],
    ["+1 (415) 555-2671", "phone"],
    ["020 7946 0958", "phone"],
    ["2024-03-05T10:30:00Z", "datetime"],
    ["March 5, 2024", "datetime"],
    ["5. März 2024", "datetime"],
    ["3:30 PM", "datetime"],
    ["tomorrow", "datetime"],
    ["#ff8800", "color"],
    ["rgba(255, 0, 0, 0.5)", "color"],
    ["cornflowerblue", "color"],
    ["123e4567-e89b-12d3-a456-426614174000", "uuid"],
    ["192.168.0.1", "ip"],
    ["::1", "ip"],
    ["/usr/local/bin/node", "path"],
    ["./relative/file.txt", "path"],
    ["report.pdf", "path"],
    ["$1,234.56", "money"],
    ["1.234,56 €", "money"],
    ["₹1,23,456", "money"],
    ["CHF 1'250.00", "money"],
    ["12 EUR", "money"],
    ["1,234", "number"],
    ["-3.14", "number"],
    ["1e10", "number"],
    ["99%", "number"],
  ];
  for (const [text, kind] of cases) {
    it(`${JSON.stringify(text)} → ${kind}`, async () => {
      const r = await parse(text, { backend: "cpu" });
      expect(r.kind).toBe(kind);
      expect(r.diagnostics.decidedBy).toMatch(/^rule:/);
    });
  }

  it("parses JSON and CSV payloads", async () => {
    expect((await parse('{"a":1}')).parsed).toEqual({ a: 1 });
    const csv = await parse('name,age\n"Doe, Jane",42\nBob,7');
    expect(csv.parsed).toEqual({
      delimiter: ",",
      header: ["name", "age"],
      rows: [
        ["Doe, Jane", "42"],
        ["Bob", "7"],
      ],
    });
    expect(parseDelimited("| a | b |\n|---|---|\n| 1 | 2 |")).toBeUndefined();
  });

  it("normalises money and numbers across locales", () => {
    expect(parseAmount("1.234,56")).toBe(1234.56);
    expect(parseAmount("1,234.56")).toBe(1234.56);
    expect(parseAmount("1 234,56")).toBe(1234.56);
    expect(parseAmount("1,5")).toBe(1.5);
    expect(parseAmount("1,234")).toBe(1234);
    expect(parseMoney("€48k")).toEqual({ amount: 48000, currency: "EUR" });
    expect(parseMoney("R$ 2.499,90")).toEqual({ amount: 2499.9, currency: "BRL" });
    expect(parseMoney("1 234,56 kr")).toEqual({ amount: 1234.56, currency: "kr" });
    expect(parseMoney("hello")).toBeUndefined();
  });

  it("normalises dates", () => {
    expect(parseDate("2024-03-05")?.iso).toBe("2024-03-05");
    expect(parseDate("March 5th, 2024 at 3pm")?.iso).toBe("2024-03-05T15:00");
    expect(parseDate("31/12/2024")?.iso).toBe("2024-12-31");
    expect(parseDate("03/04/2024")?.note).toMatch(/ambiguous/);
    expect(parseDate("next tuesday afternoon")).toEqual({});
    expect(parseDate("banana")).toBeUndefined();
  });

  it("finds regex spans in mixed text", () => {
    const text =
      "ping @sam re #412 (a3f9c2e) at 10.0.0.7, mail x@y.io, see https://a.dev/p. #wip #fff";
    const spans = ruleSpans(text).map((s) => [s.kind, text.slice(s.span[0], s.span[1])]);
    expect(spans).toEqual([
      ["mention", "@sam"],
      ["issue_ref", "#412"],
      ["commit", "a3f9c2e"],
      ["ip", "10.0.0.7"],
      ["email", "x@y.io"],
      ["url", "https://a.dev/p"],
      ["hashtag", "#wip"],
      ["color", "#fff"],
    ]);
  });
});

describe("gpu-paste model path", () => {
  it("classifies a learned kind and returns diagnostics", async () => {
    const r = await parse("- milk\n- eggs\n- bread", { backend: "cpu" });
    expect(["list", "markdown", "prose", "contact", "address", "code"]).toContain(r.kind);
    expect(r.diagnostics.backend).toBe("cpu");
    expect(r.diagnostics.kindProbabilities).toBeDefined();
    expect(r.confidence).toBeGreaterThan(0);
    expect(r.confidence).toBeLessThanOrEqual(1);
  });

  it("windows long inputs", async () => {
    const r = await parse("word ".repeat(1200), { backend: "cpu" });
    expect(r.diagnostics.windows).toBeGreaterThan(1);
  });
});
