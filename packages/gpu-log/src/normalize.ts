/** Deterministic normalizers used by every path (model, JSON, logfmt). */

export type Level = "trace" | "debug" | "info" | "warn" | "error" | "fatal";

const LEVEL_WORDS: Record<string, Level> = {
  trace: "trace",
  trce: "trace",
  verbose: "trace",
  verb: "trace",
  vrb: "trace",
  finest: "trace",
  finer: "trace",
  silly: "trace",
  t: "trace",
  v: "trace",
  debug: "debug",
  debu: "debug",
  dbug: "debug",
  dbg: "debug",
  fine: "debug",
  config: "debug",
  d: "debug",
  info: "info",
  information: "info",
  informational: "info",
  inf: "info",
  notice: "info",
  note: "info",
  log: "info",
  http: "info",
  system: "info",
  statement: "info",
  detail: "info",
  hint: "info",
  i: "info",
  n: "info",
  warn: "warn",
  warning: "warn",
  wrn: "warn",
  deprecated: "warn",
  w: "warn",
  error: "error",
  erro: "error",
  err: "error",
  "err!": "error",
  severe: "error",
  critical: "error",
  crit: "error",
  fail: "error",
  failure: "error",
  e: "error",
  fatal: "fatal",
  fata: "fatal",
  ftl: "fatal",
  emerg: "fatal",
  emergency: "fatal",
  alert: "fatal",
  panic: "fatal",
  dpanic: "fatal",
  "fatal error": "fatal",
  "parse error": "fatal",
  f: "fatal",
};

const SYSLOG_SEVERITY: Level[] = [
  "fatal",
  "fatal",
  "error",
  "error",
  "warn",
  "info",
  "info",
  "debug",
];

/** Maps any level spelling (words, single letters, syslog PRI, pino numbers) to the enum. */
export function normalizeLevel(raw: string): Level | undefined {
  let s = raw.trim().toLowerCase();
  if (s === "") return undefined;
  const pri = /^<(\d{1,3})>$/.exec(s);
  if (pri) return SYSLOG_SEVERITY[Number(pri[1]) % 8];
  if (/^\d+$/.test(s)) {
    const v = Number(s);
    if (v <= 10) return "trace";
    if (v <= 20) return "debug";
    if (v <= 30) return "info";
    if (v <= 40) return "warn";
    if (v <= 50) return "error";
    return "fatal";
  }
  // Redis markers.
  if (s === ".") return "debug";
  if (s === "-") return "trace";
  if (s === "*") return "info";
  if (s === "#") return "warn";
  s = s.replace(/^[[(<{]+|[\])>}:]+$/g, "").trim();
  return LEVEL_WORDS[s] ?? LEVEL_WORDS[s.replace(/\s+/g, " ")];
}

const MONTHS: Record<string, string> = {
  jan: "01",
  feb: "02",
  mar: "03",
  apr: "04",
  may: "05",
  jun: "06",
  jul: "07",
  aug: "08",
  sep: "09",
  sept: "09",
  oct: "10",
  nov: "11",
  dec: "12",
};

function pad2(v: string | number): string {
  return String(v).padStart(2, "0");
}

function tz(raw: string | undefined): string {
  if (!raw) return "";
  const s = raw.trim();
  if (s === "") return "";
  if (/^(z|utc|gmt)$/i.test(s)) return "Z";
  const m = /^([+-])(\d{2}):?(\d{2})?$/.exec(s);
  if (m) return `${m[1]}${m[2]}:${m[3] ?? "00"}`;
  return ""; // named zones (CET, PST) are ambiguous: leave the time unqualified
}

function frac(raw: string | undefined): string {
  if (!raw) return "";
  return `.${raw.slice(0, 9).padEnd(3, "0")}`;
}

function build(
  y: string,
  mo: string,
  d: string,
  h: string,
  mi: string,
  s: string | undefined,
  f: string | undefined,
  z: string | undefined,
): string {
  return `${y}-${pad2(mo)}-${pad2(d)}T${pad2(h)}:${pad2(mi)}:${pad2(s ?? "00")}${frac(f)}${tz(z)}`;
}

function year2(yy: string): string {
  return (Number(yy) < 70 ? "20" : "19") + yy;
}

function ampm(h: string, ap: string | undefined): string {
  if (!ap) return h;
  let v = Number(h) % 12;
  if (/pm/i.test(ap)) v += 12;
  return pad2(v);
}

const MON = "(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)";
const DOW = "(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\\s+";
const TZ = "(z|utc|gmt|[+-]\\d{2}:?\\d{2}|[+-]\\d{2}|[a-z]{3,4})?";

const PATTERNS: [RegExp, (m: RegExpExecArray) => string | undefined][] = [
  // 2024-01-15T10:30:00.123Z / 2024-01-15 10:30:00,123 +0000 / 2024-01-15T10:30:00.123456+00:00
  [
    new RegExp(
      `^(\\d{4})-(\\d{2})-(\\d{2})[t ](\\d{2}):(\\d{2})(?::(\\d{2}))?(?:[.,](\\d+))?\\s*${TZ}$`,
      "i",
    ),
    (m) => build(m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6], m[7], m[8]),
  ],
  // 2024/01/15 10:30:00.123456
  [
    /^(\d{4})\/(\d{2})\/(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?$/,
    (m) => build(m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6], m[7], undefined),
  ],
  // 15/Jan/2024:10:30:00 +0000 (apache)
  [
    new RegExp(`^(\\d{1,2})\\/${MON}\\/(\\d{4}):(\\d{2}):(\\d{2}):(\\d{2})\\s*${TZ}$`, "i"),
    (m) => build(m[3]!, MONTHS[m[2]!.toLowerCase()]!, m[1]!, m[4]!, m[5]!, m[6], undefined, m[7]),
  ],
  // Mon Jan 15 10:30:00.123456 2024 (ctime / apache error)
  [
    new RegExp(
      `^${DOW}${MON}\\s+(\\d{1,2}) (\\d{2}):(\\d{2}):(\\d{2})(?:\\.(\\d+))? (\\d{4})$`,
      "i",
    ),
    (m) => build(m[7]!, MONTHS[m[1]!.toLowerCase()]!, m[2]!, m[3]!, m[4]!, m[5], m[6], undefined),
  ],
  // Mon 2024-01-15 10:30:00 UTC (journal short-full)
  [
    new RegExp(
      `^${DOW}(\\d{4})-(\\d{2})-(\\d{2}) (\\d{2}):(\\d{2}):(\\d{2})(?:\\.(\\d+))?\\s*${TZ}$`,
      "i",
    ),
    (m) => build(m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6], m[7], m[8]),
  ],
  // 15-Jan-2024 10:30:00.123 (tomcat)
  [
    new RegExp(`^(\\d{1,2})-${MON}-(\\d{4}) (\\d{2}):(\\d{2}):(\\d{2})(?:[.,](\\d+))?$`, "i"),
    (m) => build(m[3]!, MONTHS[m[2]!.toLowerCase()]!, m[1]!, m[4]!, m[5]!, m[6], m[7], undefined),
  ],
  // 15 Jan 2024 10:30:00.123 (redis)  /  Mon, 15 Jan 2024 10:30:00 GMT (RFC 1123)
  [
    new RegExp(
      `^(?:${DOW})?(\\d{1,2}) ${MON} (\\d{4}) (\\d{2}):(\\d{2}):(\\d{2})(?:[.,](\\d+))?\\s*${TZ}$`,
      "i",
    ),
    (m) => build(m[3]!, MONTHS[m[2]!.toLowerCase()]!, m[1]!, m[4]!, m[5]!, m[6], m[7], m[8]),
  ],
  // Jan 15, 2024 10:30:00 AM (java.util.logging)
  [
    new RegExp(`^${MON} (\\d{1,2}), (\\d{4}),? (\\d{1,2}):(\\d{2}):(\\d{2})\\s*(am|pm)?$`, "i"),
    (m) =>
      build(
        m[3]!,
        MONTHS[m[1]!.toLowerCase()]!,
        m[2]!,
        ampm(m[4]!, m[7]),
        m[5]!,
        m[6],
        undefined,
        undefined,
      ),
  ],
  // 01/15/2024 10:30:00 AM (US)  /  01/15/2024, 10:30:00 AM (nest)
  [
    /^(\d{2})\/(\d{2})\/(\d{4}),? (\d{1,2}):(\d{2}):(\d{2})\s*(am|pm)?$/i,
    (m) => build(m[3]!, m[1]!, m[2]!, ampm(m[4]!, m[7]), m[5]!, m[6], undefined, undefined),
  ],
  // 15.01.2024 10:30:00 (EU)
  [
    /^(\d{2})\.(\d{2})\.(\d{4}) (\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?$/,
    (m) => build(m[3]!, m[2]!, m[1]!, m[4]!, m[5]!, m[6], m[7], undefined),
  ],
  // 24/01/15 10:30:00 (spark yy/MM/dd)
  [
    /^(\d{2})\/(\d{2})\/(\d{2}) (\d{2}):(\d{2}):(\d{2})$/,
    (m) => build(year2(m[1]!), m[2]!, m[3]!, m[4]!, m[5]!, m[6], undefined, undefined),
  ],
  // 081109 203615 (hdfs yymmdd hhmmss) / 240115 10:30:00 (mysql 5)
  [
    /^(\d{2})(\d{2})(\d{2}) (\d{2}):?(\d{2}):?(\d{2})$/,
    (m) => build(year2(m[1]!), m[2]!, m[3]!, m[4]!, m[5]!, m[6], undefined, undefined),
  ],
  // 20240115 10:30:00 / 20240115T103000Z / 20240115-10:30:00:123
  [
    /^(\d{4})(\d{2})(\d{2})[t -]?(\d{2}):?(\d{2}):?(\d{2})(?:[.,:](\d+))?\s*(z)?$/i,
    (m) => build(m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6], m[7], m[8]),
  ],
  // 2005-06-03-15.42.50.675872 (BGL)
  [
    /^(\d{4})-(\d{2})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})(?:\.(\d+))?$/,
    (m) => build(m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6], m[7], undefined),
  ],
  // epoch seconds / millis / micros / nanos, optional fraction
  [
    /^(\d{9,10})(?:\.(\d+))?$/,
    (m) => new Date(Number(m[1]) * 1000 + (m[2] ? Number(`0.${m[2]}`) * 1000 : 0)).toISOString(),
  ],
  [/^(\d{13})$/, (m) => new Date(Number(m[1])).toISOString()],
  [/^(\d{16})$/, (m) => new Date(Math.floor(Number(m[1]) / 1000)).toISOString()],
  [/^(\d{19})$/, (m) => new Date(Math.floor(Number(m[1]) / 1e6)).toISOString()],
];

/**
 * Converts a timestamp as written in a log line to ISO 8601. Returns undefined when the
 * text has no year (syslog, glog, logcat, time-only) or is not recognised. Times without a
 * zone are emitted without a suffix; named zones other than UTC/GMT are dropped.
 */
export function normalizeTimestamp(raw: string): string | undefined {
  const s = raw
    .trim()
    .replace(/^[[(]|[\])]$/g, "")
    .replace(/[,;:]$/, "");
  for (const [re, fn] of PATTERNS) {
    const m = re.exec(s);
    if (m) {
      try {
        const iso = fn(m);
        if (iso && !iso.startsWith("NaN")) return iso;
      } catch {
        return undefined;
      }
    }
  }
  return undefined;
}
