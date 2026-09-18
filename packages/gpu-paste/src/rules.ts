/**
 * Deterministic layer: whole-paste detectors and regex span detectors. Everything in this
 * file is a rule; nothing here is learned. The model (cpu.ts/gpu.ts) only runs for what
 * these rules cannot decide (see README "Rules vs. learned").
 */

export type RuleKind =
  | "json"
  | "csv"
  | "tsv"
  | "html"
  | "url"
  | "email"
  | "phone"
  | "datetime"
  | "color"
  | "uuid"
  | "jwt"
  | "ip"
  | "path"
  | "money"
  | "number";

export interface RuleMatch {
  kind: RuleKind;
  confidence: number;
  parsed: unknown;
  /** Span kind to report for the whole paste, when the paste is a single entity. */
  spanKind?: string;
  value?: string;
  note?: string;
}

export interface RuleSpan {
  kind: string;
  span: [number, number];
  value?: string;
}

// ---------------------------------------------------------------------------------------
// Currency, number and date helpers (shared by whole-paste rules and span values)

const SYMBOLS: Record<string, string> = {
  $: "USD",
  "€": "EUR",
  "£": "GBP",
  "¥": "JPY",
  "₹": "INR",
  "₽": "RUB",
  "₩": "KRW",
  "₪": "ILS",
  "₺": "TRY",
  "₫": "VND",
  "₱": "PHP",
  "฿": "THB",
  "₦": "NGN",
  "₴": "UAH",
  "₡": "CRC",
  "₲": "PYG",
  "₵": "GHS",
  US$: "USD",
  CA$: "CAD",
  C$: "CAD",
  A$: "AUD",
  AU$: "AUD",
  NZ$: "NZD",
  HK$: "HKD",
  S$: "SGD",
  MX$: "MXN",
  R$: "BRL",
  "CN¥": "CNY",
  Rs: "INR",
  "Rs.": "INR",
  "Fr.": "CHF",
  "SFr.": "CHF",
  kr: "kr",
  "kr.": "kr",
  zł: "PLN",
  Kč: "CZK",
  Ft: "HUF",
  руб: "RUB",
  "руб.": "RUB",
  円: "JPY",
  원: "KRW",
  TL: "TRY",
  R: "ZAR",
  Rp: "IDR",
};
const WORDS: Record<string, string> = {
  dollars: "USD",
  dollar: "USD",
  bucks: "USD",
  euros: "EUR",
  euro: "EUR",
  pounds: "GBP",
  quid: "GBP",
  yen: "JPY",
  rupees: "INR",
  francs: "CHF",
  pesos: "MXN",
  rand: "ZAR",
  kronor: "SEK",
  kroner: "kr",
  złotych: "PLN",
  евро: "EUR",
  рублей: "RUB",
};
const CODES = new Set(
  "USD EUR GBP JPY CNY INR AUD CAD CHF SEK NOK DKK PLN CZK HUF RUB BRL MXN ZAR KRW SGD HKD NZD TRY ILS AED SAR THB PHP IDR VND MYR NGN EGP ARS CLP COP PEN UAH RON BGN ISK TWD PKR BDT".split(
    " ",
  ),
);
const SYM_RE =
  "US\\$|CA\\$|AU\\$|NZ\\$|HK\\$|MX\\$|CN¥|SFr\\.|Fr\\.|Rs\\.?|Rp|R\\$|[ACS]\\$|[$€£¥₹₽₩₪₺₫₱฿₦₴₡₲₵]|kr\\.?|zł|Kč|Ft|руб\\.?|円|원|TL";
const NUM_RE =
  "\\d{1,2}(?:,\\d{2})+,\\d{3}(?:\\.\\d+)?|\\d{1,3}(?:[ ,.']\\d{3})+(?:[.,]\\d+)?|\\d+(?:[.,]\\d+)?";
const MULT_RE = "\\s?(?:[kK]|[mM](?:n|io\\.?)?|[bB]n|M|million|billion|thousand|Mio\\.?|Mrd\\.?)?";
const MONEY_RE = new RegExp(
  `^(?:(-|−|\\+)?(${SYM_RE}|[A-Z]{3})\\s?(-|−|\\+)?(${NUM_RE})(${MULT_RE})|(-|−|\\+)?(${NUM_RE})(${MULT_RE})\\s?(${SYM_RE}|[A-Z]{3}|${Object.keys(WORDS).join("|")}))$`,
  "u",
);

/** "1.234,56" / "1,234.56" / "1 234,56" / "1'234.50" → 1234.56. Lone 3-digit groups are thousands. */
export function parseAmount(raw: string): number | undefined {
  const s = raw.replace(/[\s']/g, "").replace(/−/g, "-");
  const m = /^([+-]?)(\d+(?:[.,]\d+)*)$/.exec(s);
  if (!m) return undefined;
  const digits = m[2]!;
  const seps = digits.match(/[.,]/g) ?? [];
  let value: string;
  if (seps.length === 0) value = digits;
  else {
    const lastSep = digits.lastIndexOf(seps[seps.length - 1]!);
    const tail = digits.slice(lastSep + 1);
    const distinct = new Set(seps).size;
    const decimal = distinct === 2 || (seps.length === 1 && tail.length !== 3);
    value = decimal
      ? `${digits.slice(0, lastSep).replace(/[.,]/g, "")}.${tail}`
      : digits.replace(/[.,]/g, "");
  }
  const n = Number(value);
  return Number.isFinite(n) ? (m[1] === "-" ? -n : n) : undefined;
}

function multiplier(s: string): number {
  const t = s.trim().toLowerCase();
  if (!t) return 1;
  if (t === "k" || t === "thousand") return 1e3;
  if (t.startsWith("b") || t.startsWith("mrd")) return 1e9;
  return 1e6;
}

export function parseMoney(text: string): { amount: number; currency: string } | undefined {
  const m = MONEY_RE.exec(text.trim());
  if (!m) return undefined;
  const sign =
    m[1] === "-" || m[1] === "−" || m[3] === "-" || m[3] === "−" || m[6] === "-" || m[6] === "−"
      ? -1
      : 1;
  const cur = (m[2] ?? m[9])!;
  const num = (m[4] ?? m[7])!;
  const mult = multiplier(m[5] ?? m[8] ?? "");
  let currency = SYMBOLS[cur] ?? WORDS[cur.toLowerCase()];
  if (!currency) {
    if (/^[A-Z]{3}$/.test(cur) && CODES.has(cur)) currency = cur;
    else return undefined;
  }
  const amount = parseAmount(num);
  if (amount === undefined) return undefined;
  return { amount: sign * amount * mult, currency };
}

const MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split(" ");
const MONTH_RE =
  "(jan(?:uary|vier|uar)?|feb(?:ruary|ruar)?|f[ée]vrier|m[aä]r(?:ch|s|z|zo)?|apr(?:il)?|avril|abril|ma[iy]o?|jun(?:e|i|io)?|juin|jul(?:y|i|io)?|juillet|aug(?:ust)?|ao[uû]t|agosto|sep(?:t(?:ember|iembre|embre)?)?|oct(?:ober|obre|ubre)?|okt(?:ober)?|nov(?:ember|embre|iembre)?|dec(?:ember)?|d[ée]cembre|dez(?:ember)?|diciembre)\\.?";
const DAY_RE = "(?:mon|tue|wed|thu|fri|sat|sun)(?:day|sday|nesday|rsday|urday)?";
const TIME_RE =
  "(\\d{1,2})(?::(\\d{2}))?(?::(\\d{2}))?\\s?([ap]\\.?m\\.?)?(?:\\s?(?:[A-Z]{2,5}|UTC[+-]\\d{1,2}|[+-]\\d{2}:?\\d{2}))?";

function monthIndex(name: string): number {
  const n = name.toLowerCase().replace(".", "");
  const table: Record<string, number> = {
    mär: 2,
    marz: 2,
    märz: 2,
    mars: 2,
    marzo: 2,
    avril: 3,
    abril: 3,
    mai: 4,
    mayo: 4,
    juin: 5,
    juni: 5,
    junio: 5,
    juillet: 6,
    juli: 6,
    julio: 6,
    aout: 7,
    août: 7,
    agosto: 7,
    okt: 9,
    oktober: 9,
    dez: 11,
    dezember: 11,
    décembre: 11,
    decembre: 11,
    diciembre: 11,
    fevrier: 1,
    février: 1,
    enero: 0,
    janvier: 0,
    januar: 0,
  };
  if (n in table) return table[n]!;
  const i = MONTHS.indexOf(n.slice(0, 3));
  return i;
}

const pad = (n: number | string) => String(n).padStart(2, "0");

function isoTime(h: string, m?: string, s?: string, ap?: string): string {
  let hour = Number(h);
  const a = ap?.toLowerCase().replace(/\./g, "");
  if (a === "pm" && hour < 12) hour += 12;
  if (a === "am" && hour === 12) hour = 0;
  return `${pad(hour)}:${m ?? "00"}${s ? `:${s}` : ""}`;
}

/** Whole-string date/time recogniser. Returns an ISO-ish normalisation when unambiguous. */
export function parseDate(text: string): { iso?: string; note?: string } | undefined {
  const s = text.trim();
  let m =
    /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$/.exec(
      s,
    );
  if (m) {
    return {
      iso: m[4]
        ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}${m[6] ? `:${m[6]}` : ""}${m[7] ?? ""}`
        : `${m[1]}-${m[2]}-${m[3]}`,
    };
  }
  m = /^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$/.exec(s);
  if (m) {
    return { iso: `${m[1]}-${pad(+m[2]!)}-${pad(+m[3]!)}` };
  }
  m = /^(\d{1,2})([/.-])(\d{1,2})\2(\d{2,4})$/.exec(s);
  if (m) {
    const a = +m[1]!;
    const b = +m[3]!;
    let y = +m[4]!;
    if (y < 100) y += 2000;
    if (a > 12 && b > 12) return undefined;
    const dayFirst = a > 12 || (b <= 12 && m[2] !== "/");
    const [mo, d] = dayFirst ? [b, a] : [a, b];
    const note =
      a <= 12 && b <= 12
        ? "day/month order ambiguous; assumed " + (dayFirst ? "day-first" : "month-first")
        : undefined;
    return note ? { iso: `${y}-${pad(mo)}-${pad(d)}`, note } : { iso: `${y}-${pad(mo)}-${pad(d)}` };
  }
  const named = new RegExp(
    `^(?:${DAY_RE},?\\s+)?(?:(?:the\\s+)?(\\d{1,2})(?:st|nd|rd|th|\\.|er)?\\s+(?:of\\s+|de\\s+)?${MONTH_RE}(?:\\s+(?:de\\s+)?(\\d{4}))?|${MONTH_RE}\\s+(\\d{1,2})(?:st|nd|rd|th)?,?(?:\\s+(\\d{4}))?|${MONTH_RE}\\s+(\\d{4}))(?:,?\\s+(?:at\\s+|um\\s+|à\\s+)?${TIME_RE})?$`,
    "iu",
  );
  m = named.exec(s);
  if (m) {
    const day = m[1] ?? m[5];
    const month = m[2] ?? m[4] ?? m[7];
    const year = m[3] ?? m[6] ?? m[8];
    const mi = monthIndex(month!);
    if (mi < 0) return undefined;
    let iso = year ? `${year}-${pad(mi + 1)}` : `--${pad(mi + 1)}`;
    if (day) iso += `-${pad(+day)}`;
    if (m[9]) iso += `T${isoTime(m[9], m[10], m[11], m[12])}`;
    return { iso };
  }
  const rfc = new RegExp(
    `^(?:${DAY_RE},?\\s+)?(\\d{1,2})\\s+${MONTH_RE}\\s+(\\d{4})\\s+(\\d{2}):(\\d{2})(?::(\\d{2}))?\\s*(?:[+-]\\d{4}|[A-Z]{2,4})?$`,
    "i",
  );
  m = rfc.exec(s);
  if (m) {
    return {
      iso: `${m[3]}-${pad(monthIndex(m[2]!) + 1)}-${pad(+m[1]!)}T${m[4]}:${m[5]}${m[6] ? `:${m[6]}` : ""}`,
    };
  }
  m = new RegExp(`^${TIME_RE}$`).exec(s);
  if (m && (m[2] || m[4])) {
    return { iso: `T${isoTime(m[1]!, m[2], m[3], m[4])}` };
  }
  const relative = new RegExp(
    `^(?:today|tomorrow|yesterday|tonight|noon|midnight|now|(?:next|last|this)\\s+(?:week|month|year|quarter)|(?:(?:next|last|this)\\s+)?${DAY_RE}(?:\\s+(?:morning|afternoon|evening|night))?(?:,?\\s+(?:at\\s+)?${TIME_RE})?|(?:tomorrow|today|tonight)\\s+(?:morning|afternoon|evening|night|(?:at\\s+)?${TIME_RE})|in\\s+\\d+\\s+(?:minutes?|hours?|days?|weeks?|months?|years?)|\\d+\\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\\s+ago|end\\s+of\\s+(?:day|week|month|year|${MONTH_RE})|(?:early|mid|late|mid-)\\s?${MONTH_RE}|eod|eow|eom|q[1-4]\\s?\\d{4}|fy\\s?\\d{2,4}|${MONTH_RE}\\s+\\d{1,2}\\s?[-–]\\s?\\d{1,2}(?:,?\\s+\\d{4})?)$`,
    "iu",
  );
  if (relative.test(s)) return {};
  return undefined;
}

// ---------------------------------------------------------------------------------------
// Colors

const NAMED: Record<string, string> = {};
for (const [
  name,
  hex,
] of "aliceblue f0f8ff antiquewhite faebd7 aqua 00ffff aquamarine 7fffd4 azure f0ffff beige f5f5dc bisque ffe4c4 black 000000 blanchedalmond ffebcd blue 0000ff blueviolet 8a2be2 brown a52a2a burlywood deb887 cadetblue 5f9ea0 chartreuse 7fff00 chocolate d2691e coral ff7f50 cornflowerblue 6495ed cornsilk fff8dc crimson dc143c cyan 00ffff darkblue 00008b darkcyan 008b8b darkgoldenrod b8860b darkgray a9a9a9 darkgreen 006400 darkgrey a9a9a9 darkkhaki bdb76b darkmagenta 8b008b darkolivegreen 556b2f darkorange ff8c00 darkorchid 9932cc darkred 8b0000 darksalmon e9967a darkseagreen 8fbc8f darkslateblue 483d8b darkslategray 2f4f4f darkslategrey 2f4f4f darkturquoise 00ced1 darkviolet 9400d3 deeppink ff1493 deepskyblue 00bfff dimgray 696969 dimgrey 696969 dodgerblue 1e90ff firebrick b22222 floralwhite fffaf0 forestgreen 228b22 fuchsia ff00ff gainsboro dcdcdc ghostwhite f8f8ff gold ffd700 goldenrod daa520 gray 808080 green 008000 greenyellow adff2f grey 808080 honeydew f0fff0 hotpink ff69b4 indianred cd5c5c indigo 4b0082 ivory fffff0 khaki f0e68c lavender e6e6fa lavenderblush fff0f5 lawngreen 7cfc00 lemonchiffon fffacd lightblue add8e6 lightcoral f08080 lightcyan e0ffff lightgoldenrodyellow fafad2 lightgray d3d3d3 lightgreen 90ee90 lightgrey d3d3d3 lightpink ffb6c1 lightsalmon ffa07a lightseagreen 20b2aa lightskyblue 87cefa lightslategray 778899 lightslategrey 778899 lightsteelblue b0c4de lightyellow ffffe0 lime 00ff00 limegreen 32cd32 linen faf0e6 magenta ff00ff maroon 800000 mediumaquamarine 66cdaa mediumblue 0000cd mediumorchid ba55d3 mediumpurple 9370db mediumseagreen 3cb371 mediumslateblue 7b68ee mediumspringgreen 00fa9a mediumturquoise 48d1cc mediumvioletred c71585 midnightblue 191970 mintcream f5fffa mistyrose ffe4e1 moccasin ffe4b5 navajowhite ffdead navy 000080 oldlace fdf5e6 olive 808000 olivedrab 6b8e23 orange ffa500 orangered ff4500 orchid da70d6 palegoldenrod eee8aa palegreen 98fb98 paleturquoise afeeee palevioletred db7093 papayawhip ffefd5 peachpuff ffdab9 peru cd853f pink ffc0cb plum dda0dd powderblue b0e0e6 purple 800080 rebeccapurple 663399 red ff0000 rosybrown bc8f8f royalblue 4169e1 saddlebrown 8b4513 salmon fa8072 sandybrown f4a460 seagreen 2e8b57 seashell fff5ee sienna a0522d silver c0c0c0 skyblue 87ceeb slateblue 6a5acd slategray 708090 slategrey 708090 snow fffafa springgreen 00ff7f steelblue 4682b4 tan d2b48c teal 008080 thistle d8bfd8 tomato ff6347 turquoise 40e0d0 violet ee82ee wheat f5deb3 white ffffff whitesmoke f5f5f5 yellow ffff00 yellowgreen 9acd32"
  .split(" ")
  .reduce<[string, string][]>((acc, v, i, arr) => (i % 2 ? acc : [...acc, [v, arr[i + 1]!]]), [])) {
  NAMED[name] = hex;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

export interface Color {
  hex: string;
  rgb: [number, number, number];
  alpha?: number;
}

export function parseColor(text: string): Color | undefined {
  const s = text.trim();
  let m = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(s);
  if (m) {
    let h = m[1]!.toLowerCase();
    if (h.length <= 4) h = [...h].map((c) => c + c).join("");
    const rgb: [number, number, number] = [
      Number.parseInt(h.slice(0, 2), 16),
      Number.parseInt(h.slice(2, 4), 16),
      Number.parseInt(h.slice(4, 6), 16),
    ];
    const out: Color = { hex: `#${h.slice(0, 6)}`, rgb };
    if (h.length === 8) out.alpha = Number.parseInt(h.slice(6, 8), 16) / 255;
    return out;
  }
  m =
    /^(rgba?|hsla?)\(\s*([\d.]+)(%?)\s*[, ]\s*([\d.]+)%?\s*[, ]\s*([\d.]+)%?\s*(?:[,/]\s*([\d.]+)(%?)\s*)?\)$/i.exec(
      s,
    );
  if (m) {
    const a = m[6] !== undefined ? Number(m[6]) / (m[7] ? 100 : 1) : undefined;
    let rgb: [number, number, number];
    if (m[1]!.toLowerCase().startsWith("hsl"))
      rgb = hslToRgb(Number(m[2]), Number(m[4]) / 100, Number(m[5]) / 100);
    else {
      const scale = m[3] ? 2.55 : 1;
      rgb = [Number(m[2]) * scale, Number(m[4]) * scale, Number(m[5]) * scale].map((v) =>
        Math.max(0, Math.min(255, Math.round(v))),
      ) as [number, number, number];
    }
    const out: Color = { hex: `#${rgb.map((v) => pad(v.toString(16))).join("")}`, rgb };
    if (a !== undefined) out.alpha = a;
    return out;
  }
  const hex = NAMED[s.toLowerCase()];
  if (hex) return parseColor(`#${hex}`);
  return undefined;
}

// ---------------------------------------------------------------------------------------
// Network / identifier rules

const UUID_RE = /^\{?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\}?$/i;
const IPV4_RE = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?(?:\/(\d{1,2}))?$/;

function ipv4(s: string): { address: string; port?: number; prefix?: number } | undefined {
  const m = IPV4_RE.exec(s);
  if (!m) return undefined;
  const octets = m.slice(1, 5).map(Number);
  if (octets.some((o) => o > 255)) return undefined;
  const out: { address: string; port?: number; prefix?: number } = { address: octets.join(".") };
  if (m[5]) out.port = Number(m[5]);
  if (m[6]) out.prefix = Number(m[6]);
  return out;
}

function ipv6(s: string): { address: string; prefix?: number } | undefined {
  const m = /^\[?([0-9a-f:]+)\]?(?:\/(\d{1,3}))?$/i.exec(s);
  if (!m) return undefined;
  const body = m[1]!;
  const parts = body.split("::");
  if (parts.length > 2 || body.includes(":::")) return undefined;
  const groups = parts.flatMap((p) => (p ? p.split(":") : []));
  if (groups.some((g) => g.length === 0 || g.length > 4)) return undefined;
  if (parts.length === 2 ? groups.length > 7 : groups.length !== 8) return undefined;
  const out: { address: string; prefix?: number } = { address: body.toLowerCase() };
  if (m[2]) out.prefix = Number(m[2]);
  return out;
}

function base64urlJson(s: string): unknown {
  const b64 = s
    .replace(/-/g, "+")
    .replace(/_/g, "/")
    .padEnd(Math.ceil(s.length / 4) * 4, "=");
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

const TLD =
  "com|org|net|io|dev|co|ai|app|edu|gov|mil|int|uk|de|fr|jp|ca|au|in|nl|se|ch|it|es|br|me|xyz|info|biz|eu|us|tv|cc|ly|sh|so|to|gg|nz|ie|no|dk|fi|pl|ru|cn|kr|mx|za|be|at|pt|cz|hu|ro|tr|il|ar|cl|sg|hk|tw|id|ph|vn|th|my|ae|sa|ng|ke|eg|ua|gr|link|site|online|store|tech|cloud|page|blog|design|studio|agency|digital|wiki|news|email|zone|fyi|pro|lol|wtf|today|world";
const BARE_URL = new RegExp(
  `^(?:www\\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+(?:${TLD})(?::\\d{2,5})?(?:[/?#][^\\s]*)?$`,
  "i",
);
const EMAIL_RE =
  /^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$/i;

interface UrlParts {
  href: string;
  protocol: string;
  host: string;
  pathname: string;
  search: string;
  hash: string;
}

function parseUrl(s: string): UrlParts | undefined {
  let href = s;
  if (/^[a-z][a-z0-9+.-]*:/i.test(s)) {
    if (!/^(?:https?|ftp|ftps|wss?|mailto|tel|file|ssh|git|s3|gs):/i.test(s)) return undefined;
  } else if (BARE_URL.test(s)) href = `https://${s}`;
  else return undefined;
  if (/\s/.test(s)) return undefined;
  try {
    const u = new URL(href);
    if (!u.hostname && !/^(mailto|tel|file):/i.test(href)) return undefined;
    return {
      href: u.href,
      protocol: u.protocol.replace(":", ""),
      host: u.host,
      pathname: u.pathname,
      search: u.search,
      hash: u.hash,
    };
  } catch {
    return undefined;
  }
}

const PHONE_RE =
  /^(\+|00)?[\d\s().\-–]{5,24}(?:\s?(?:x|ext\.?|extension|,\s?ext\.?|#)\s?(\d{1,5}))?$/i;

export function parsePhone(
  text: string,
): { digits: string; e164?: string; extension?: string } | undefined {
  const s = text.trim();
  const m = PHONE_RE.exec(s);
  if (!m) return undefined;
  const main = m[2]
    ? s.slice(0, s.length - m[2].length).replace(/(?:x|ext\.?|extension|#|,)\s*$/i, "")
    : s;
  const digits = main.replace(/\D/g, "");
  if (digits.length < 7 || digits.length > 15) return undefined;
  if (!/[\s().\-–+]/.test(main) && !(digits.length >= 10 && /^[2-9]/.test(digits)))
    return undefined;
  const out: { digits: string; e164?: string; extension?: string } = { digits };
  if (m[1]) out.e164 = `+${digits.replace(/^00/, "")}`;
  if (m[2]) out.extension = m[2];
  return out;
}

const NUMBER_RE =
  /^[+-−]?(?:\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:[ ,.']\d{3})+|\d+)(?:[.,]\d+)?(?:\s?%|[eE][+-]?\d+)?$|^0x[0-9a-f]+$|^0b[01]+$/i;

export function parseNumber(text: string): number | undefined {
  const s = text.trim().replace(/−/, "-");
  if (!NUMBER_RE.test(s)) return undefined;
  if (/^0x/i.test(s)) return Number.parseInt(s.slice(2), 16);
  if (/^0b/i.test(s)) return Number.parseInt(s.slice(2), 2);
  const pct = s.endsWith("%");
  const exp = /[eE]([+-]?\d+)$/.exec(s);
  const body = s.replace(/\s?%$/, "").replace(/[eE][+-]?\d+$/, "");
  let n = parseAmount(body);
  if (n === undefined) return undefined;
  if (exp) n *= 10 ** Number(exp[1]);
  return pct ? n / 100 : n;
}

// ---------------------------------------------------------------------------------------
// Structured text rules

export function parseDelimited(
  text: string,
): { delimiter: string; rows: string[][]; header?: string[] } | undefined {
  const lines = text.replace(/\r\n?/g, "\n").replace(/\n+$/, "").split("\n");
  if (lines.length < 2 || lines.some((l) => /^\s*\|/.test(l))) return undefined;
  for (const delimiter of ["\t", ",", ";", "|"]) {
    const rows: string[][] = [];
    let ok = true;
    for (const line of lines) {
      const cells: string[] = [];
      let cell = "";
      let quoted = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i]!;
        if (quoted) {
          if (ch === '"') {
            if (line[i + 1] === '"') {
              cell += '"';
              i++;
            } else quoted = false;
          } else cell += ch;
        } else if (ch === '"' && cell === "") quoted = true;
        else if (ch === delimiter) {
          cells.push(cell);
          cell = "";
        } else cell += ch;
      }
      if (quoted) {
        ok = false;
        break;
      }
      cells.push(cell);
      rows.push(cells);
    }
    const width = rows[0]!.length;
    if (!ok || width < 2 || rows.some((r) => r.length !== width)) continue;
    // Cells are values, not sentences: prose with one ";" per line is not a table.
    const cells = rows.flat();
    const shortCells = cells.filter(
      (c) => c.trim().split(/\s+/).length <= 4 && !/[.!?]$/.test(c.trim()),
    ).length;
    if (delimiter !== "\t" && shortCells < cells.length * 0.6) continue;
    const out: { delimiter: string; rows: string[][]; header?: string[] } = { delimiter, rows };
    const first = rows[0]!;
    const numeric = (c: string) => parseNumber(c) !== undefined;
    if (
      !first.some(numeric) &&
      rows.slice(1).some((r) => r.some(numeric)) &&
      first.every((c) => c.trim())
    ) {
      out.header = first;
      out.rows = rows.slice(1);
    }
    return out;
  }
  return undefined;
}

const HTML_PAIR = /<([a-z][\w-]*)(?:\s[^<>]*)?>[\s\S]*?<\/\1\s*>/i;
const HTML_VOID = /<(?:br|hr|img|input|meta|link)\b[^<>]*\/?>/i;

export function parseHtml(text: string): { tags: string[]; text: string } | undefined {
  const s = text.trim();
  const isDoc = /^<!doctype\s+html/i.test(s) || /<html[\s>]/i.test(s);
  if (!isDoc && !HTML_PAIR.test(s) && !(HTML_VOID.test(s) && /^<[^<>]+>$/.test(s)))
    return undefined;
  const tags = [...new Set([...s.matchAll(/<([a-z][\w-]*)/gi)].map((m) => m[1]!.toLowerCase()))];
  return {
    tags,
    text: s
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim(),
  };
}

const WIN_PATH = /^(?:[a-zA-Z]:\\|\\\\)[^\n<>"|?*]*$/;
const POSIX_PATH = /^(?:\/|~\/|\.{1,2}\/)[^\s]*$/;
const REL_PATH = /^(?:[\w@.~-]+\/)+[\w@.~-]*$/;
const FILE_NAME = /^[\w-][\w. -]*\.[a-z0-9]{1,8}$/i;
const EXTENSIONS = new Set(
  "md txt pdf png jpg jpeg gif svg webp csv tsv json yaml yml toml xml html htm css scss js mjs cjs ts tsx jsx py rb go rs java kt swift c h cpp hpp cs php sh zip tar gz bz2 7z mp3 mp4 mov wav avi doc docx xls xlsx ppt pptx key numbers pages sql db sqlite log ini cfg conf env lock exe dll so dylib app dmg iso ttf otf woff woff2 ipynb parquet".split(
    " ",
  ),
);

export function parsePath(
  text: string,
): { segments: string[]; basename: string; extension?: string; absolute: boolean } | undefined {
  const s = text.trim();
  const isWin = WIN_PATH.test(s);
  const isPosix = POSIX_PATH.test(s) || REL_PATH.test(s);
  if (!isWin && !isPosix) {
    const m = FILE_NAME.exec(s);
    if (!m || !EXTENSIONS.has(s.slice(s.lastIndexOf(".") + 1).toLowerCase())) return undefined;
  }
  const segments = s.split(/[\\/]+/).filter(Boolean);
  const basename = segments[segments.length - 1] ?? "";
  const dot = basename.lastIndexOf(".");
  const out: { segments: string[]; basename: string; extension?: string; absolute: boolean } = {
    segments,
    basename,
    absolute: isWin || s.startsWith("/"),
  };
  if (dot > 0) out.extension = basename.slice(dot + 1).toLowerCase();
  return out;
}

// ---------------------------------------------------------------------------------------
// Whole-paste detector (order matters: most specific and fully validated first).

export function detectWhole(text: string): RuleMatch | undefined {
  const s = text.trim();
  if (!s) return undefined;
  const m = UUID_RE.exec(s);
  if (m) {
    const uuid = m[1]!.toLowerCase();
    return {
      kind: "uuid",
      confidence: 1,
      parsed: { uuid, version: Number.parseInt(uuid[14]!, 16) },
      spanKind: "uuid",
      value: uuid,
    };
  }
  if (/^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*$/.test(s)) {
    try {
      const [h, p] = s.split(".");
      const header = base64urlJson(h!) as Record<string, unknown>;
      if (header && typeof header === "object" && "alg" in header) {
        return { kind: "jwt", confidence: 1, parsed: { header, payload: base64urlJson(p!) } };
      }
    } catch {}
  }
  const v4 = ipv4(s);
  if (v4)
    return {
      kind: "ip",
      confidence: 1,
      parsed: { version: 4, ...v4 },
      spanKind: "ip",
      value: v4.address,
    };
  const v6 = s.includes(":") ? ipv6(s) : undefined;
  if (v6)
    return {
      kind: "ip",
      confidence: 0.95,
      parsed: { version: 6, ...v6 },
      spanKind: "ip",
      value: v6.address,
    };
  const color = parseColor(s);
  if (color && !/^#\d{3,4}$/.test(s))
    return { kind: "color", confidence: 1, parsed: color, spanKind: "color", value: color.hex };
  if (EMAIL_RE.test(s)) {
    const [local, domain] = s.split("@");
    return {
      kind: "email",
      confidence: 1,
      parsed: { address: s.toLowerCase(), local, domain: domain!.toLowerCase() },
      spanKind: "email",
      value: s.toLowerCase(),
    };
  }
  const url = parseUrl(s);
  if (url)
    return {
      kind: "url",
      confidence: /^[a-z][a-z0-9+.-]*:/i.test(s) ? 1 : 0.9,
      parsed: url,
      spanKind: "url",
      value: url.href,
    };
  const money = parseMoney(s);
  if (money)
    return {
      kind: "money",
      confidence: 0.95,
      parsed: money,
      spanKind: "money",
      value: `${money.amount} ${money.currency}`,
    };
  const date = parseDate(s);
  if (date) {
    const out: RuleMatch = {
      kind: "datetime",
      confidence: date.iso ? 0.95 : 0.85,
      parsed: { ...date, raw: s },
      spanKind: "date",
    };
    if (date.iso) out.value = date.iso;
    if (date.note) out.note = date.note;
    return out;
  }
  const number = parseNumber(s);
  if (number !== undefined && !(/^\d{10,11}$/.test(s) && /^[2-9]/.test(s))) {
    return { kind: "number", confidence: 1, parsed: number, value: String(number) };
  }
  const phone = parsePhone(s);
  if (phone)
    return {
      kind: "phone",
      confidence: /^\d+$/.test(s) ? 0.6 : 0.9,
      parsed: phone,
      spanKind: "phone",
      value: phone.e164 ?? phone.digits,
    };
  const path = parsePath(s);
  if (path) return { kind: "path", confidence: path.absolute ? 0.95 : 0.8, parsed: path };
  if (s[0] === "{" || s[0] === "[") {
    try {
      return { kind: "json", confidence: 1, parsed: JSON.parse(s) };
    } catch {}
  }
  const html = parseHtml(s);
  if (html) return { kind: "html", confidence: 0.9, parsed: html };
  const table = parseDelimited(text);
  if (table)
    return { kind: table.delimiter === "\t" ? "tsv" : "csv", confidence: 0.85, parsed: table };
  return undefined;
}

// ---------------------------------------------------------------------------------------
// Regex span detectors inside mixed text. UTF-16 offsets.

const SPAN_PATTERNS: [string, RegExp, (m: RegExpExecArray) => string | undefined][] = [
  [
    "email",
    /[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+/gi,
    (m) => m[0].toLowerCase(),
  ],
  [
    "url",
    /(?:https?|ftp|wss?):\/\/[^\s<>"'`)\]}]+|(?<![\w@/.])www\.[^\s<>"'`)\]}]+/gi,
    (m) => parseUrl(m[0].replace(/[.,;:!?]+$/, ""))?.href,
  ],
  [
    "uuid",
    /(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])/gi,
    (m) => m[0].toLowerCase(),
  ],
  ["ip", /(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])/g, (m) => ipv4(m[0])?.address],
  [
    "color",
    /(?<![\w#])#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3,4})(?![\w-])|\b(?:rgba?|hsla?)\([^)]*\)/gi,
    (m) => parseColor(m[0])?.hex,
  ],
  [
    "date",
    /(?<!\d)\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)/g,
    (m) => parseDate(m[0])?.iso,
  ],
  ["issue_ref", /(?<![\w#/-])(?:#\d{1,7}|(?:GH|[A-Z]{2,10})-\d{1,7})(?![\w-])/g, (m) => m[0]],
  ["hashtag", /(?<![\w&#])#[\p{L}_][\p{L}\p{N}_]*/gu, (m) => m[0].slice(1)],
  [
    "mention",
    /(?<![\w@.])@[a-z0-9_][a-z0-9_.-]{1,38}(?![\w@])/gi,
    (m) => m[0].slice(1).replace(/\.$/, ""),
  ],
  [
    "commit",
    /(?<![\w/-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![\w-])/gi,
    (m) => m[0].toLowerCase(),
  ],
];

/** Deterministic spans. Colors beat issue refs on "#abc"; issue refs beat colors on "#123". */
export function ruleSpans(text: string): RuleSpan[] {
  const out: RuleSpan[] = [];
  const taken: [number, number][] = [];
  const overlaps = (s: number, e: number) => taken.some(([ts, te]) => s < te && e > ts);
  for (const [kind, re, toValue] of SPAN_PATTERNS) {
    re.lastIndex = 0;
    for (;;) {
      const m = re.exec(text);
      if (!m) break;
      let raw = m[0];
      if (kind === "url" || kind === "email") raw = raw.replace(/[.,;:!?]+$/, "");
      if (kind === "color" && /^#\d{3,4}$/.test(raw)) continue; // "#123" is an issue ref, not a colour
      if (kind === "ip" && !ipv4(raw)) continue;
      const start = m.index;
      const end = start + raw.length;
      if (overlaps(start, end)) continue;
      taken.push([start, end]);
      const span: RuleSpan = { kind, span: [start, end] };
      const v = toValue(m);
      if (v !== undefined) span.value = v;
      out.push(span);
    }
  }
  return out.sort((a, b) => a.span[0] - b.span[0]);
}
