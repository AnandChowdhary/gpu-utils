import { tokenize } from "@gpu-utils/runtime";
import { describe, expect, it } from "vitest";
import {
  bioTransitions,
  compileLine,
  frameLanguage,
  mergeTwoLineFrames,
  spansFromTags,
} from "../src/decode.ts";
import type { LogLine } from "../src/types.ts";

const LABELS = [
  "O",
  "B-TS",
  "I-TS",
  "B-LEVEL",
  "I-LEVEL",
  "B-SOURCE",
  "I-SOURCE",
  "B-THREAD",
  "I-THREAD",
  "B-KEY",
  "I-KEY",
  "B-VALUE",
  "I-VALUE",
  "B-MSG",
  "I-MSG",
  "B-FN",
  "I-FN",
  "B-FILE",
  "I-FILE",
  "B-LINE",
  "I-LINE",
  "B-COL",
  "I-COL",
];
const id = (l: string) => LABELS.indexOf(l);

function tagsFor(text: string, roles: Record<string, string>): Int32Array {
  // roles: substring -> role; tokens inside a substring get B/I tags.
  const tokens = tokenize(text);
  const path = new Int32Array(tokens.length);
  for (const [sub, role] of Object.entries(roles)) {
    const start = text.indexOf(sub);
    const end = start + sub.length;
    let first = true;
    tokens.forEach((t, i) => {
      if (t.start >= start && t.end <= end) {
        path[i] = id(`${first ? "B" : "I"}-${role}`);
        first = false;
      }
    });
  }
  return path;
}

describe("decode", () => {
  it("forbids O -> I-X transitions", () => {
    const t = bioTransitions(LABELS);
    expect(t[0 * LABELS.length + id("I-TS")]).toBe(-Infinity);
    expect(t[id("B-TS") * LABELS.length + id("I-TS")]).toBe(0);
    expect(t[id("B-TS") * LABELS.length + id("I-MSG")]).toBe(-Infinity);
  });

  it("groups BIO tags into spans", () => {
    const path = Int32Array.from([
      id("B-TS"),
      id("I-TS"),
      0,
      id("B-KEY"),
      id("B-VALUE"),
      id("I-VALUE"),
    ]);
    expect(spansFromTags(path, LABELS)).toEqual([
      { role: "TS", start: 0, end: 2 },
      { role: "KEY", start: 3, end: 4 },
      { role: "VALUE", start: 4, end: 6 },
    ]);
  });

  it("compiles kv pairs, message, level and timestamp", () => {
    const text = '2024-01-15T10:30:00Z WARN app: disk almost full path="/var/log" pct=91';
    const path = tagsFor(text, {
      "2024-01-15T10:30:00Z": "TS",
      WARN: "LEVEL",
      app: "SOURCE",
      "disk almost full": "MSG",
      path: "KEY",
      "/var/log": "VALUE",
      pct: "KEY",
      "91": "VALUE",
    });
    const line = compileLine(
      text,
      tokenize(text),
      path,
      Float32Array.from([1, 0, 0]),
      LABELS,
      3,
      [10, 20],
    );
    expect(line).toEqual({
      line: 3,
      span: [10, 20],
      kind: "entry",
      timestamp: { text: "2024-01-15T10:30:00Z", iso: "2024-01-15T10:30:00Z" },
      level: "warn",
      source: "app",
      message: "disk almost full",
      kv: [
        { key: "path", value: "/var/log" },
        { key: "pct", value: "91" },
      ],
    });
  });

  it("merges JSON payload messages and names access-log values", () => {
    const text = '2024-01-15T10:30:00Z stdout F {"level":"error","msg":"boom","id":7}';
    const path = tagsFor(text, {
      "2024-01-15T10:30:00Z": "TS",
      '{"level":"error","msg":"boom","id":7}': "MSG",
    });
    const line = compileLine(text, tokenize(text), path, Float32Array.from([1, 0, 0]), LABELS, 0, [
      0,
      text.length,
    ]);
    expect(line.level).toBe("error");
    expect(line.message).toBe("boom");
    expect(line.kv).toEqual([{ key: "id", value: "7" }]);

    const access = '"GET / HTTP/1.1" 200 512 "-" "curl"';
    const p2 = tagsFor(access, {
      "GET / HTTP/1.1": "MSG",
      "200": "VALUE",
      "512": "VALUE",
      "-": "VALUE",
      curl: "VALUE",
    });
    const l2 = compileLine(access, tokenize(access), p2, Float32Array.from([1, 0, 0]), LABELS, 0, [
      0,
      access.length,
    ]);
    expect(l2.kv).toEqual([
      { key: "status", value: "200" },
      { key: "bytes", value: "512" },
      { key: "referrer", value: "-" },
      { key: "user_agent", value: "curl" },
    ]);
  });

  it("detects frame languages", () => {
    expect(frameLanguage("\tat a.b.C.d(C.java:1)", { function: "a.b.C.d", file: "C.java" })).toBe(
      "java",
    );
    expect(frameLanguage('  File "x.py", line 3, in f', { file: "x.py" })).toBe("python");
    expect(frameLanguage("    at f (/a/b.js:1:2)", { file: "/a/b.js" })).toBe("javascript");
    expect(frameLanguage("\t/go/src/main.go:12 +0x1d", { file: "/go/src/main.go" })).toBe("go");
    expect(
      frameLanguage("   0: std::panicking::begin_panic", {
        function: "std::panicking::begin_panic",
      }),
    ).toBe("rust");
    expect(
      frameLanguage("   at A.B.C(Int32 x) in C:\\a.cs:line 3", {
        function: "A.B.C",
        file: "C:\\a.cs",
      }),
    ).toBe("csharp");
    expect(frameLanguage("\tfrom /a/b.rb:3:in `x'", { file: "/a/b.rb" })).toBe("ruby");
    expect(frameLanguage("#0 /a/b.php(3): f()", { file: "/a/b.php" })).toBe("php");
  });

  it("merges Go/Rust two-line frames", () => {
    const lines: LogLine[] = [
      {
        line: 0,
        span: [0, 1],
        kind: "frame",
        kv: [],
        frame: { function: "main.main", language: "go" },
      },
      { line: 1, span: [2, 3], kind: "frame", kv: [], frame: { file: "/app/main.go", line: 12 } },
    ];
    mergeTwoLineFrames(lines);
    expect(lines[1]!.frame).toEqual({
      file: "/app/main.go",
      line: 12,
      function: "main.main",
      language: "go",
    });
  });
});
