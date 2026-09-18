import { describe, expect, it } from "vitest";
import { fastPath, splitLogfmt } from "../src/fastpath.ts";
import { featurize } from "../src/features.ts";
import { parse, splitLines } from "../src/index.ts";
import { normalizeLevel, normalizeTimestamp } from "../src/normalize.ts";

describe("splitLines", () => {
  it("handles LF, CRLF and a missing final newline", () => {
    expect(splitLines("a\r\nb\n\nc")).toEqual([
      [0, 1],
      [3, 4],
      [5, 5],
      [6, 7],
    ]);
    expect(splitLines("")).toEqual([]);
    expect(splitLines("x\n")).toEqual([[0, 1]]);
  });
});

describe("featurize", () => {
  it("emits FEATURE_COUNT ids per token", () => {
    const f = featurize("2024-01-15 INFO hello");
    expect(f.rows).toHaveLength(f.tokens.length);
    for (const row of f.rows) expect(row).toHaveLength(9);
  });
});

describe("normalizeLevel", () => {
  it("maps spellings, letters, PRI and numbers", () => {
    expect(normalizeLevel("WARNING")).toBe("warn");
    expect(normalizeLevel("[error]")).toBe("error");
    expect(normalizeLevel("INFO:")).toBe("info");
    expect(normalizeLevel("E")).toBe("error");
    expect(normalizeLevel("<134>")).toBe("info");
    expect(normalizeLevel("<0>")).toBe("fatal");
    expect(normalizeLevel("30")).toBe("info");
    expect(normalizeLevel("50")).toBe("error");
    expect(normalizeLevel("Fatal error")).toBe("fatal");
    expect(normalizeLevel("banana")).toBeUndefined();
  });
});

describe("normalizeTimestamp", () => {
  it("converts full-date formats to ISO 8601", () => {
    expect(normalizeTimestamp("2024-01-15T10:30:00.123Z")).toBe("2024-01-15T10:30:00.123Z");
    expect(normalizeTimestamp("2024-01-15 10:30:00,123")).toBe("2024-01-15T10:30:00.123");
    expect(normalizeTimestamp("2024/01/15 10:30:00")).toBe("2024-01-15T10:30:00");
    expect(normalizeTimestamp("15/Jan/2024:10:30:00 +0000")).toBe("2024-01-15T10:30:00+00:00");
    expect(normalizeTimestamp("Mon Jan 15 10:30:00.123456 2024")).toBe(
      "2024-01-15T10:30:00.123456",
    );
    expect(normalizeTimestamp("01/15/2024 10:30:00 PM")).toBe("2024-01-15T22:30:00");
    expect(normalizeTimestamp("24/01/15 10:30:00")).toBe("2024-01-15T10:30:00");
    expect(normalizeTimestamp("081109 203615")).toBe("2008-11-09T20:36:15");
    expect(normalizeTimestamp("1705314600")).toBe("2024-01-15T10:30:00.000Z");
    expect(normalizeTimestamp("1705314600123")).toBe("2024-01-15T10:30:00.123Z");
    expect(normalizeTimestamp("Jan 15, 2024 10:30:00 AM")).toBe("2024-01-15T10:30:00");
    expect(normalizeTimestamp("2024-01-15 10:30:00.123 +05:30")).toBe(
      "2024-01-15T10:30:00.123+05:30",
    );
    expect(normalizeTimestamp("2016-09-28 04:30:30,")).toBe("2016-09-28T04:30:30");
  });
  it("returns undefined without a year", () => {
    expect(normalizeTimestamp("Jan 15 10:30:00")).toBeUndefined();
    expect(normalizeTimestamp("10:30:00.123")).toBeUndefined();
    expect(normalizeTimestamp("0115 10:30:00.123456")).toBeUndefined();
  });
});

describe("fast paths", () => {
  it("parses JSON lines deterministically", () => {
    const r = fastPath(
      '{"level":30,"time":1705314600123,"pid":42,"hostname":"h","msg":"hello","reqId":"abc"}',
    );
    expect(r?.path).toBe("json");
    expect(r?.fields.level).toBe("info");
    expect(r?.fields.timestamp?.iso).toBe("2024-01-15T10:30:00.123Z");
    expect(r?.fields.message).toBe("hello");
    expect(r?.fields.thread).toBe("42");
    expect(r?.fields.kv).toEqual([
      { key: "hostname", value: "h" },
      { key: "reqId", value: "abc" },
    ]);
  });
  it("parses pure logfmt and rejects mixed lines", () => {
    expect(
      splitLogfmt('time=2024-01-15T10:30:00Z level=info msg="hello world" n=3 empty='),
    ).toEqual([
      { key: "time", value: "2024-01-15T10:30:00Z" },
      { key: "level", value: "info" },
      { key: "msg", value: "hello world" },
      { key: "n", value: "3" },
      { key: "empty", value: "" },
    ]);
    expect(splitLogfmt("2024-01-15 INFO msg=hi")).toBeUndefined();
    expect(splitLogfmt("plain text")).toBeUndefined();
    const r = fastPath('ts=2024-01-15T10:30:00Z level=warn msg="disk full" path=/var');
    expect(r?.path).toBe("logfmt");
    expect(r?.fields.level).toBe("warn");
    expect(r?.fields.message).toBe("disk full");
    expect(r?.fields.kv).toEqual([{ key: "path", value: "/var" }]);
  });
  it("unwraps CRI-prefixed JSON lines", () => {
    const r = fastPath('2024-01-15T10:30:00.123456789Z stdout F {"level":"error","msg":"boom"}');
    expect(r?.path).toBe("json");
    expect(r?.fields.timestamp?.text).toBe("2024-01-15T10:30:00.123456789Z");
    expect(r?.fields.level).toBe("error");
  });
});

describe("parse", () => {
  it("returns one entry per line with spans and kinds", async () => {
    const text =
      'x=1 y=2\n\n{"msg":"j"}\n2024-01-15 10:30:00,123 [main] INFO  com.example.Foo - Started\n';
    const r = await parse(text, { backend: "cpu" });
    expect(r.lines).toHaveLength(4);
    expect(r.lines.map((l) => l.kind)).toEqual(["entry", "blank", "entry", "entry"]);
    expect(r.lines[0]!.kv).toEqual([
      { key: "x", value: "1" },
      { key: "y", value: "2" },
    ]);
    expect(r.lines[2]!.message).toBe("j");
    expect(r.lines[3]!.span).toEqual([21, 83]);
    expect(text.slice(...r.lines[3]!.span)).toBe(
      "2024-01-15 10:30:00,123 [main] INFO  com.example.Foo - Started",
    );
    expect(r.stats).toMatchObject({ json: 1, logfmt: 1, model: 1, blank: 1, backend: "cpu" });
  });

  it("extracts fields from common formats via the model", async () => {
    const text = [
      "2024-01-15 10:30:00,123 [main] INFO  com.example.Foo - Started server on port 8080",
      "Jan 15 10:30:00 myhost sshd[1234]: Accepted publickey for alice from 10.0.0.1 port 22 ssh2",
      '10.0.0.1 - - [15/Jan/2024:10:30:00 +0000] "GET /index.html HTTP/1.1" 200 1234 "-" "curl/8.4.0"',
      "\tat com.example.Foo.bar(Foo.java:42)",
      '  File "/app/main.py", line 12, in <module>',
      "    at fetchUser (/app/src/users.js:10:15)",
    ].join("\n");
    const r = await parse(text, { backend: "cpu" });
    const [a, b, c, d, e, f] = r.lines;
    expect(a!.timestamp?.iso).toBe("2024-01-15T10:30:00.123");
    expect(a!.level).toBe("info");
    expect(a!.source).toBe("com.example.Foo");
    expect(a!.thread).toBe("main");
    expect(a!.message).toBe("Started server on port 8080");
    expect(b!.timestamp?.text).toBe("Jan 15 10:30:00");
    expect(b!.host).toBe("myhost");
    expect(b!.source).toBe("sshd");
    expect(b!.thread).toBe("1234");
    expect(c!.timestamp?.iso).toBe("2024-01-15T10:30:00+00:00");
    expect(c!.message).toBe("GET /index.html HTTP/1.1");
    expect(c!.kv.find((p) => p.key === "status")?.value).toBe("200");
    expect(d!.kind).toBe("frame");
    expect(d!.frame).toMatchObject({
      function: "com.example.Foo.bar",
      file: "Foo.java",
      line: 42,
      language: "java",
    });
    expect(e!.frame).toMatchObject({
      file: "/app/main.py",
      line: 12,
      function: "<module>",
      language: "python",
    });
    expect(f!.frame).toMatchObject({
      function: "fetchUser",
      file: "/app/src/users.js",
      line: 10,
      column: 15,
      language: "javascript",
    });
  });

  it("labels the host of a cluster RAS line", async () => {
    const { lines } = await parse(
      "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
      { backend: "cpu" },
    );
    expect(lines[0]).toMatchObject({
      host: "R02-M1-N0-C:J12-U11",
      source: "KERNEL",
      level: "info",
      message: "instruction cache parity error corrected",
    });
  });

  it("can disable fast paths and batches many lines per forward pass", async () => {
    const line = "2024-01-15T10:30:00Z INFO app: hello k=v";
    const text = Array.from({ length: 40 }, () => line).join("\n");
    const r = await parse(text, { backend: "cpu", batchTokens: 64 });
    expect(r.lines).toHaveLength(40);
    for (const l of r.lines) expect(l.level).toBe("info");
    const r2 = await parse("a=1 b=2", { backend: "cpu", fastPaths: false });
    expect(r2.stats.model).toBe(1);
  });
});

describe("memoization", () => {
  it("reuses model output for repeated templates and matches the unmemoized result", async () => {
    const text = [
      "2024-01-15 10:30:00,123 WARN app: user 42 logged in",
      "2024-02-17 11:45:09,001 WARN app: user 77 logged in",
      "\tat a.b.C.d(C.java:12)",
      "\tat a.b.C.d(C.java:99)",
    ].join("\n");
    const a = await parse(text, { backend: "cpu" });
    const b = await parse(text, { backend: "cpu", memoize: false });
    expect(a.stats.memoized).toBe(2);
    expect(b.stats.memoized).toBe(0);
    expect(a.lines).toEqual(b.lines);
    expect(a.lines[1]!.level).toBe("warn");
    expect(a.lines[3]!.frame?.line).toBe(99);
  });
});
