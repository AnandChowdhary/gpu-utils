# gpu-log

Universal log line parser: timestamps, levels, sources, threads, key=value pairs, messages
and stack frames from any text log format.

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-log
```

```ts
import { parse } from "gpu-log";

const { lines } = await parse(`
2024-01-15 10:30:00,123 [main] INFO  com.example.Foo - Started server on port 8080 region=eu-west-1
Jan 15 10:30:01 web-01 sshd[1234]: Accepted publickey for alice from 10.0.0.9 port 51234 ssh2
{"level":30,"time":1705314601123,"msg":"request completed","reqId":"abc","responseTime":12}
java.lang.IllegalStateException: pool exhausted
\tat com.example.Foo.bar(Foo.java:42)
`);

lines[1];
// {
//   line: 1, span: [1, 98], kind: "entry",
//   timestamp: { text: "2024-01-15 10:30:00,123", iso: "2024-01-15T10:30:00.123" },
//   level: "info", thread: "main", source: "com.example.Foo",
//   message: "Started server on port 8080",
//   kv: [{ key: "region", value: "eu-west-1" }]
// }
lines[2].source;        // "sshd"      lines[2].thread; // "1234"
lines[3].level;         // "info"      (JSON lines never touch the model)
lines[4].kind;          // "continuation"
lines[5].frame;         // { function: "com.example.Foo.bar", file: "Foo.java", line: 42, language: "java" }
```

Every line of the input becomes one `LogLine`:

```ts
interface LogLine {
  line: number;                       // zero-based
  span: [number, number];             // UTF-16 offsets in the input, line break excluded
  kind: "entry" | "continuation" | "frame" | "blank";
  timestamp?: { text: string; iso?: string };
  level?: "trace" | "debug" | "info" | "warn" | "error" | "fatal";
  source?: string;                    // logger / process / component
  thread?: string;                    // thread name, pid, request id
  kv: { key: string; value: string }[];
  message?: string;
  frame?: { function?: string; file?: string; line?: number; column?: number; language?: string };
}
```

`parse(text, { backend, fastPaths, batchTokens, memoize })` returns `{ lines, stats }`; `stats`
reports how many lines took each path and which backend ran. `backend: "auto"` (default) uses
WebGPU when it exists and there are at least 256 model tokens, otherwise the CPU reference path.
`normalizeTimestamp()` and `normalizeLevel()` are exported for reuse.

## How it works

Blank lines, JSON objects and pure logfmt lines are handled by deterministic code and never
reach the model (JSON keys such as `time`, `level`, `msg`, `logger`, `pid` map onto the
fields; the rest become `kv`). Every other line is split into character-class runs by the
shared gpu-utils tokenizer, and each token gets nine sparse hashed feature ids (word hash,
3-char prefix hash, shape, first/last character, length, position from the start and end,
column). No vocabulary is learned. A 150K-parameter residual dilated 1-D CNN (embedding 32 →
hidden 64, five blocks with dilations 1, 2, 4, 8, 16, kernel 3, int6 weights) tags every
token with one of 23 BIO labels over the roles TS, LEVEL, SOURCE, THREAD, KEY, VALUE, MSG,
FN, FILE, LINE, COL, and a second head, mean-pooled over the line, decides whether the line
is an entry, a continuation or a stack frame. Lines are packed back to back into one flat
token sequence with a line-id array, so one GPU dispatch tags up to 131,072 tokens (thousands
of lines) and the convolutions never read across line boundaries. A Viterbi pass enforces the
BIO grammar on the CPU, and a deterministic compiler turns spans into the typed record:
timestamp → ISO 8601, level spellings (`WARNING`, `W`, `<134>`, `30`, `wrn`) → the six-value
enum, unkeyed values after an HTTP request line → `status`/`bytes`/`referrer`/`user_agent`,
frame language from the frame's shape, Go/Rust two-line frames merged. Repeated templates
are memoised within a call: the hashing collapses digits, so lines that differ only in numbers
have identical features and reuse the model output exactly.

## Size and speed

The size budget for this package is 120 KB Brotli instead of the default 40 KB: the task
needs long-context tagging over noisy, long lines (dilated-CNN family, 150K parameters) and a
timestamp/level/frame compiler that covers ~45 timestamp spellings and 8 stack-trace dialects.

| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | __SIZE__ |
| Parameters | 150,042 (int6) |
| CPU reference path, 10 MB file, memoisation off | __CPU_RAW__ |
| CPU reference path, 10 MB file, memoisation on | __CPU_MEMO__ |
| JSON / logfmt fast path only | __FAST__ |
| WebGPU path (design estimate, see MODEL_CARD.md) | __GPU__ |

Numbers from `node packages/gpu-log/bench/throughput.mjs` on this box (Node 24, one core).

## Limitations

- The model sees one line at a time. A plain line with no timestamp or level is an `entry`;
  it cannot know that it continues the previous entry unless the line itself looks like a
  continuation (indent, `Caused by:`, `... 12 more`, exception headers).
- Hostnames are not extracted (`source` is the process/logger; the pid is `thread`).
- `timestamp.iso` is only set when the text contains a year: syslog, glog, logcat and
  time-only stamps keep `text` only. Named zones other than UTC/GMT are dropped.
- Key=value pairs are recognised when they follow the message or a structured prefix;
  `word=value` inside prose is treated as message text.
- Single-line frames from languages outside the eight trained dialects usually get FILE/LINE
  but `language` may be missing or wrong.
- WebGPU output is checked against the CPU path in a browser (`bench/gpu-parity.mjs`),
  not in CI: Node has no WebGPU device.

See [MODEL_CARD.md](./MODEL_CARD.md) for evaluation numbers on generated, hand-written and
real (Loghub) logs.

## Training

```bash
cd packages/gpu-log/training
uv sync
uv run python -m gpu_log.data       # generate + cache 160K synthetic lines, print samples
uv run python -m gpu_log.train      # ~12 min on 2 CPU threads, QAT int6
uv run python -m gpu_log.export     # write ../model/{manifest.json,weights.txt,fixtures.json}
uv run python -m gpu_log.evaluate   # held-out, unfamiliar and Loghub sets
```
