import type { Level } from "./normalize.ts";

export type { Level } from "./normalize.ts";

export type LineKind = "entry" | "continuation" | "frame" | "blank";

export interface LogTimestamp {
  /** The timestamp exactly as written. */
  text: string;
  /** ISO 8601 when the text carries a full date; omitted for year-less formats (syslog). */
  iso?: string;
}

export interface LogKv {
  key: string;
  value: string;
}

export interface LogFrame {
  function?: string;
  file?: string;
  line?: number;
  column?: number;
  /** java | python | javascript | go | rust | csharp | ruby | php, when recognisable. */
  language?: string;
}

export interface LogLine {
  /** Zero-based line number. */
  line: number;
  /** UTF-16 offsets of the line in the input, half-open, excluding the line break. */
  span: [number, number];
  kind: LineKind;
  timestamp?: LogTimestamp;
  level?: Level;
  source?: string;
  thread?: string;
  kv: LogKv[];
  message?: string;
  frame?: LogFrame;
}

export interface LogParseResult {
  lines: LogLine[];
  /** How many lines went through each path (useful for tuning and tests). */
  stats: {
    json: number;
    logfmt: number;
    /** Lines that needed the model (including memoised repeats). */
    model: number;
    /** Model lines answered from an identical feature sequence seen earlier in the call. */
    memoized: number;
    blank: number;
    backend: "cpu" | "webgpu";
  };
}
