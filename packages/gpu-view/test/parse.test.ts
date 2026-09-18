import { describe, expect, it } from "vitest";
import { parse, type Schema, SchemaError } from "../src/index.ts";
import { strip } from "./helpers.ts";

const sales: Schema = {
  fields: [
    { name: "revenue", kind: "number", aliases: ["sales", "amount"] },
    { name: "region", kind: "enum", values: ["EMEA", "APAC", "Americas"] },
    { name: "closed_at", kind: "date", aliases: ["closed", "close date"] },
    { name: "owner", kind: "text", aliases: ["rep"] },
  ],
};
const issues: Schema = {
  fields: [
    { name: "status", kind: "enum", values: ["open", "in progress", "closed"] },
    { name: "assignee", kind: "text", aliases: ["assigned", "assigned to"] },
    {
      name: "priority",
      kind: "enum",
      values: ["low", "medium", "high", "urgent"],
    },
    { name: "created", kind: "date" },
  ],
};
const customers: Schema = {
  fields: [
    { name: "name", kind: "text" },
    { name: "country", kind: "enum", values: ["Germany", "France", "Spain"] },
    { name: "orders", kind: "number", aliases: ["order count"] },
    { name: "signed_up", kind: "date", aliases: ["signup"] },
    { name: "active", kind: "boolean" },
  ],
};
const orders: Schema = {
  fields: [
    { name: "order_value", kind: "number", aliases: ["value", "total"] },
    { name: "placed_at", kind: "date", aliases: ["placed", "order date"] },
    { name: "channel", kind: "enum", values: ["web", "store"] },
  ],
};
const now = "2026-09-17T12:00:00Z";

describe("gpu-view examples", () => {
  it("total revenue by region this quarter, top 10, as a bar chart", async () => {
    const spec = await parse("total revenue by region this quarter, top 10, as a bar chart", {
      schema: sales,
      now,
      backend: "cpu",
    });
    expect(spec.diagnostics).toEqual([]);
    expect(strip(spec, false)).toEqual({
      filters: [
        {
          field: "closed_at",
          op: "between",
          value: ["2026-07-01", "2026-09-30"],
        },
      ],
      sort: [],
      groupBy: [{ field: "region" }],
      aggregate: [{ fn: "sum", field: "revenue" }],
      limit: 10,
      chart: "bar",
    });
    expect(spec.filters[0]!.span).toEqual({ start: 24, end: 36 }); // "this quarter"
  });
  it("open issues assigned to me sorted by priority", async () => {
    const spec = await parse("open issues assigned to me sorted by priority", {
      schema: issues,
      now,
      backend: "cpu",
    });
    expect(strip(spec, false)).toEqual({
      filters: [
        { field: "status", op: "eq", value: "open" },
        { field: "assignee", op: "eq", value: "me" },
      ],
      sort: [{ field: "priority", dir: "asc" }],
      groupBy: [],
      aggregate: [],
    });
  });
  it("customers in Germany or France with more than 5 orders", async () => {
    const spec = await parse("customers in Germany or France with more than 5 orders", {
      schema: customers,
      now,
      backend: "cpu",
    });
    expect(strip(spec, false)).toEqual({
      filters: [
        { field: "country", op: "in", value: ["Germany", "France"] },
        { field: "orders", op: "gt", value: 5 },
      ],
      sort: [],
      groupBy: [],
      aggregate: [],
    });
  });
  it("average order value per month last year, line chart", async () => {
    const spec = await parse("average order value per month last year, line chart", {
      schema: orders,
      now,
      backend: "cpu",
    });
    expect(strip(spec, false)).toEqual({
      filters: [
        {
          field: "placed_at",
          op: "between",
          value: ["2025-01-01", "2025-12-31"],
        },
      ],
      sort: [],
      groupBy: [{ field: "placed_at" }],
      aggregate: [{ fn: "avg", field: "order_value" }],
      chart: "line",
      granularity: "month",
    });
  });
  it("emits a diagnostic instead of guessing an unknown field", async () => {
    const text = "customers with more than 5 widgets";
    const spec = await parse(text, { schema: customers, now, backend: "cpu" });
    // No filter may be invented for "widgets"; the number is reported, not attached to a guessed field.
    expect(spec.filters.filter((f) => f.value === 5)).toEqual([]);
    expect(spec.diagnostics.length).toBeGreaterThan(0);
    const d = spec.diagnostics[0]!;
    expect(["unknown_field", "unresolved_value"]).toContain(d.code);
    expect(["widgets", "5"]).toContain(text.slice(d.span.start, d.span.end));
  });
  it("tolerates plurals, aliases and one-character typos in field names", async () => {
    const spec = await parse("active custmers with orderz over 10k, signup after 2024", {
      schema: customers,
      now,
      backend: "cpu",
    });
    expect(spec.filters).toContainEqual(
      expect.objectContaining({ field: "orders", op: "gt", value: 10000 }),
    );
    expect(spec.filters).toContainEqual(
      expect.objectContaining({
        field: "signed_up",
        op: "gt",
        value: "2024-12-31",
      }),
    );
    expect(spec.filters).toContainEqual(
      expect.objectContaining({ field: "active", op: "is_true" }),
    );
  });
  it("throws SchemaError on a bad schema", async () => {
    await expect(
      parse("x", {
        schema: { fields: [{ name: "a", kind: "blob" as "text" }] },
      }),
    ).rejects.toBeInstanceOf(SchemaError);
  });
  it("returns an empty spec for empty input", async () => {
    const spec = await parse("", { schema: customers, backend: "cpu" });
    expect(spec.filters).toEqual([]);
    expect(spec.tokens).toEqual([]);
  });
});
