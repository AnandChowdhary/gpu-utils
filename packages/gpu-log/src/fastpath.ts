/**
 * Deterministic parsers that run before the model: JSON lines and pure logfmt lines.
 * They also provide the field mapping reused when the model finds a JSON payload in a message.
 */
import { type Level, normalizeLevel, normalizeTimestamp } from "./normalize.ts";
import type { LogKv, LogLine } from "./types.ts";

const TIME_KEYS = new Set([
  "@timestamp",
  "timestamp",
  "time",
  "ts",
  "t",
  "@t",
  "datetime",
  "date",
  "eventtime",
  "event_time",
  "asctime",
  "logtime",
  "_source_realtime_timestamp",
]);
const LEVEL_KEYS = new Set([
  "level",
  "lvl",
  "severity",
  "levelname",
  "log.level",
  "@l",
  "priority",
  "loglevel",
  "log_level",
  "severity_text",
  "at",
]);
const MSG_KEYS = new Set(["message", "msg", "@m", "log", "text", "event", "body", "@message"]);
const SOURCE_KEYS = new Set([
  "logger",
  "logger_name",
  "loggername",
  "name",
  "source",
  "component",
  "module",
  "app",
  "service",
  "target",
  "category",
  "channel",
  "syslog_identifier",
  "_comm",
  "sourcecontext",
]);
const THREAD_KEYS = new Set([
  "thread",
  "thread_name",
  "threadname",
  "threadid",
  "thread_id",
  "pid",
  "tid",
  "process",
  "processid",
  "goroutine",
  "_pid",
]);

type Fields = Pick<LogLine, "timestamp" | "level" | "source" | "thread" | "message" | "kv">;

function stringify(v: unknown): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Maps a flat object's keys onto the LogLine fields; unknown keys become kv pairs. */
export function fieldsFromObject(obj: Record<string, unknown>, into: Fields): void {
  for (const [key, value] of Object.entries(obj)) {
    const k = key.toLowerCase();
    if (
      TIME_KEYS.has(k) &&
      !into.timestamp &&
      (typeof value === "string" || typeof value === "number")
    ) {
      const text = String(value);
      const iso = normalizeTimestamp(text);
      into.timestamp = iso ? { text, iso } : { text };
      continue;
    }
    if (
      LEVEL_KEYS.has(k) &&
      !into.level &&
      (typeof value === "string" || typeof value === "number")
    ) {
      const level = normalizeLevel(String(value));
      if (level) {
        into.level = level;
        continue;
      }
    }
    if (MSG_KEYS.has(k) && into.message === undefined && typeof value === "string") {
      into.message = value;
      continue;
    }
    if (SOURCE_KEYS.has(k) && !into.source && typeof value === "string" && value !== "") {
      into.source = value;
      continue;
    }
    if (
      THREAD_KEYS.has(k) &&
      !into.thread &&
      (typeof value === "string" || typeof value === "number")
    ) {
      into.thread = String(value);
      continue;
    }
    into.kv.push({ key, value: stringify(value) });
  }
}

/** Tries the JSON path: the trimmed line must be one JSON object. */
export function parseJsonLine(text: string): Fields | undefined {
  const t = text.trim();
  if (t.length < 2 || t.charCodeAt(0) !== 123 /* { */ || t.charCodeAt(t.length - 1) !== 125 /* } */)
    return undefined;
  let obj: unknown;
  try {
    obj = JSON.parse(t);
  } catch {
    return undefined;
  }
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return undefined;
  const fields: Fields = { kv: [] };
  fieldsFromObject(obj as Record<string, unknown>, fields);
  return fields;
}

const LOGFMT_PAIR = /([A-Za-z0-9_.@$/:-]+)=("(?:[^"\\]|\\.)*"|[^\s"]*)/y;
const KEY_START = /[A-Za-z0-9_.@$/:-]/;

/** Splits a pure logfmt line into pairs; undefined if anything else is on the line. */
export function splitLogfmt(text: string): LogKv[] | undefined {
  const pairs: LogKv[] = [];
  let i = 0;
  const n = text.length;
  while (i < n) {
    const c = text[i]!;
    if (c === " " || c === "\t") {
      i++;
      continue;
    }
    if (!KEY_START.test(c)) return undefined;
    LOGFMT_PAIR.lastIndex = i;
    const m = LOGFMT_PAIR.exec(text);
    if (!m || m.index !== i) return undefined;
    let value = m[2]!;
    if (value.startsWith('"')) value = value.slice(1, -1).replace(/\\(.)/g, "$1");
    pairs.push({ key: m[1]!, value });
    i = LOGFMT_PAIR.lastIndex;
    if (i < n && text[i] !== " " && text[i] !== "\t") return undefined;
  }
  return pairs.length > 0 ? pairs : undefined;
}

/** Tries the logfmt path: every whitespace-separated token must be key=value. */
export function parseLogfmtLine(text: string): Fields | undefined {
  const pairs = splitLogfmt(text);
  if (!pairs) return undefined;
  const fields: Fields = { kv: [] };
  const obj: Record<string, unknown> = {};
  // Keep duplicate keys as kv entries rather than overwriting.
  for (const p of pairs) {
    if (p.key in obj) fields.kv.push(p);
    else obj[p.key] = p.value;
  }
  const extra = fields.kv;
  fields.kv = [];
  fieldsFromObject(obj, fields);
  fields.kv.push(...extra);
  return fields;
}

/** Prefixes that wrap another line: CRI (`ts stream flag `), kubectl --timestamps, docker -t. */
const WRAPPER =
  /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})) (?:(?:stdout|stderr) [FP] )?/;

/**
 * Runs both fast paths on a line, also when it is wrapped by a container-runtime timestamp
 * prefix. Returns undefined when the line needs the model.
 */
export function fastPath(text: string): { fields: Fields; path: "json" | "logfmt" } | undefined {
  let inner = text;
  let prefixTs: string | undefined;
  const w = WRAPPER.exec(text);
  if (w && w[0].length < text.length) {
    inner = text.slice(w[0].length);
    prefixTs = w[1];
  }
  const json = parseJsonLine(inner);
  if (json) {
    if (prefixTs && !json.timestamp) {
      const iso = normalizeTimestamp(prefixTs);
      json.timestamp = iso ? { text: prefixTs, iso } : { text: prefixTs };
    }
    return { fields: json, path: "json" };
  }
  if (prefixTs) return undefined; // wrapped non-JSON lines carry their own prefix roles: model
  const logfmt = parseLogfmtLine(inner);
  if (logfmt) return { fields: logfmt, path: "logfmt" };
  return undefined;
}

export type { Level };
