"""Synthetic data generator for gpu-paste.

Every example is (text, kind, spans). `kind` is one of the six learned kinds or None when
the example only trains the span head (JSON/CSV/HTML/log pastes with entities inside, or a
bare entity whose kind the decoder derives deterministically). Spans are UTF-16 half-open
offsets tagged with one of the six learned span kinds.

Sources (see MODEL_CARD.md and THIRD_PARTY_NOTICES.md):
  - Faker (MIT): names, streets, cities, postcodes and company names in 25+ locales.
  - US Census Bureau 1990 name files (public domain): extra first/last names (downloaded).
  - SEC EDGAR company_tickers.json (public domain): real company names (downloaded).
Downloads are optional; without network the generator falls back to Faker only.

`uv run python -m gpu_paste.data` prints a sample; `generate()` is used by train.py.
"""

from __future__ import annotations

import json
import random
import string
import sys
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from faker import Faker

LEARNED_KINDS = ["address", "contact", "prose", "list", "code", "markdown"]
SPAN_KINDS = ["person", "company", "address", "date", "money", "phone"]
LABELS = ["O"] + [f"{p}-{k}" for k in SPAN_KINDS for p in ("B", "I")]

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"

LOCALES = [
    "en_US", "en_US", "en_US", "en_GB", "en_GB", "en_CA", "en_AU", "en_IN", "en_NZ", "en_IE",
    "de_DE", "de_AT", "de_CH", "fr_FR", "fr_CA", "es_ES", "es_MX", "it_IT", "nl_NL", "pt_BR",
    "pt_PT", "sv_SE", "pl_PL", "tr_TR", "cs_CZ", "da_DK", "no_NO", "fi_FI", "ja_JP", "ko_KR",
    "zh_CN", "ru_RU", "id_ID", "hu_HU", "ro_RO",
]
COUNTRY = {
    "en_US": "USA", "en_GB": "United Kingdom", "en_CA": "Canada", "en_AU": "Australia",
    "en_IN": "India", "en_NZ": "New Zealand", "en_IE": "Ireland", "de_DE": "Germany",
    "de_AT": "Austria", "de_CH": "Switzerland", "fr_FR": "France", "fr_CA": "Canada",
    "es_ES": "Spain", "es_MX": "Mexico", "it_IT": "Italy", "nl_NL": "Netherlands",
    "pt_BR": "Brazil", "pt_PT": "Portugal", "sv_SE": "Sweden", "pl_PL": "Poland",
    "tr_TR": "Turkey", "cs_CZ": "Czech Republic", "da_DK": "Denmark", "no_NO": "Norway",
    "fi_FI": "Finland", "ja_JP": "Japan", "ko_KR": "South Korea", "zh_CN": "China",
    "ru_RU": "Russia", "id_ID": "Indonesia", "hu_HU": "Hungary", "ro_RO": "Romania",
}


def _utf16(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


@dataclass
class Example:
    text: str
    kind: str | None
    spans: list[tuple[int, int, str]] = field(default_factory=list)


class Builder:
    """Accumulates text while recording UTF-16 spans of tagged pieces."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[tuple[int, int, str]] = []
        self.offset = 0

    def add(self, text: str, kind: str | None = None) -> "Builder":
        if kind and text.strip():
            lead = len(text) - len(text.lstrip())
            trail = len(text) - len(text.rstrip())
            start = self.offset + _utf16(text[:lead])
            end = self.offset + _utf16(text) - _utf16(text[len(text) - trail :] if trail else "")
            self.spans.append((start, end, kind))
        self.parts.append(text)
        self.offset += _utf16(text)
        return self

    def text(self) -> str:
        return "".join(self.parts)

    def build(self, kind: str | None) -> Example:
        return Example(self.text(), kind, list(self.spans))


# --------------------------------------------------------------------------------------
# Optional real-world word lists (public domain)


def _download(name: str, url: str) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gpu-utils-training gpu-paste@anandchowdhary.com", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310
            data = r.read().decode("utf-8", errors="replace")
        path.write_text(data, encoding="utf-8")
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[data] could not download {name}: {e}", file=sys.stderr)
        return None


def load_wordlists(offline: bool = False) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {"first": [], "last": [], "company": []}
    if offline:
        return lists
    base = "https://www2.census.gov/topics/genealogy/1990surnames/"
    for key, fname in (("last", "dist.all.last"), ("first", "dist.female.first"), ("first", "dist.male.first")):
        text = _download(fname, base + fname)
        if text:
            names = [line.split()[0].capitalize() for line in text.splitlines() if line.strip()]
            lists[key].extend(names[:1500])
    tickers = _download("company_tickers.json", "https://www.sec.gov/files/company_tickers.json")
    if tickers:
        try:
            rows = json.loads(tickers)
            names = [v["title"] for v in rows.values()]
            lists["company"] = [n.title() if n.isupper() else n for n in names[:4000]]
        except Exception:  # noqa: BLE001
            pass
    return lists


# --------------------------------------------------------------------------------------


class Gen:
    def __init__(self, seed: int, offline: bool = False) -> None:
        self.rng = random.Random(seed)
        self.fakers: dict[str, Faker] = {}
        for loc in sorted(set(LOCALES)):
            f = Faker(loc)
            f.seed_instance(seed * 1000 + zlib.crc32(loc.encode()) % 1000)
            self.fakers[loc] = f
        self.words = load_wordlists(offline)

    # -- helpers --------------------------------------------------------------------
    def r(self) -> random.Random:
        return self.rng

    def pick(self, xs):
        return self.rng.choice(xs)

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def faker(self, english_bias: float = 0.55) -> tuple[str, Faker]:
        if self.chance(english_bias):
            loc = self.pick(["en_US", "en_US", "en_GB", "en_CA", "en_AU", "en_IN", "en_NZ", "en_IE"])
        else:
            loc = self.pick(LOCALES)
        return loc, self.fakers[loc]

    def digits(self, n: int) -> str:
        return "".join(self.pick(string.digits) for _ in range(n))

    # -- entities ---------------------------------------------------------------------
    def person(self) -> str:
        loc, f = self.faker()
        if self.words["first"] and self.chance(0.3):
            first = self.pick(self.words["first"])
            last = self.pick(self.words["last"]) if self.words["last"] else f.last_name()
            name = f"{first} {last}"
        elif self.chance(0.75):
            name = f"{f.first_name()} {f.last_name()}"
        else:
            name = f.name()
        roll = self.rng.random()
        if roll < 0.08:
            name = f"{self.pick(['Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Dr', 'Mx.'])} {name}"
        elif roll < 0.14:
            name = f"{name}{self.pick([', PhD', ' Jr.', ', MD', ' III', ', CPA', ' Sr.'])}"
        elif roll < 0.2:
            parts = name.split()
            if len(parts) >= 2:
                name = f"{parts[0][0]}. {parts[-1]}"
        elif roll < 0.25:
            parts = name.split()
            if len(parts) >= 2:
                name = f"{parts[0]} {self.pick(string.ascii_uppercase)}. {parts[-1]}"
        elif roll < 0.29:
            parts = name.split()
            if len(parts) == 2:
                name = f"{parts[1]}, {parts[0]}"
        elif roll < 0.33:
            name = name.upper()
        elif roll < 0.36:
            name = name.lower()
        return name

    def company(self) -> str:
        roll = self.rng.random()
        if self.words["company"] and roll < 0.3:
            return self.pick(self.words["company"])
        if roll < 0.7:
            loc, f = self.faker()
            return f.company()
        adj = self.pick(["Blue", "Northwind", "Acme", "Bright", "Apex", "Summit", "Nova", "Pioneer", "Cedar", "Quantum", "Atlas", "Orbit", "Lumen", "Vertex", "Harbor", "Silver", "Alpine", "Delta", "Golden", "Iron"])
        noun = self.pick(["Labs", "Systems", "Technologies", "Analytics", "Consulting", "Logistics", "Foods", "Studio", "Partners", "Software", "Robotics", "Media", "Capital", "Health", "Energy", "Designs", "Networks", "Ventures", "Bakery", "Motors"])
        suffix = self.pick(["", "", "", " Inc.", " Inc", " LLC", " Ltd", " Ltd.", " GmbH", " AG", " S.A.", " B.V.", " Pty Ltd", " Co.", " Corp.", " plc", " SAS", " S.r.l.", " Oy", " AB", " & Co.", " Group", " Holdings"])
        return f"{adj} {noun}{suffix}"

    def email(self, name: str | None = None) -> str:
        loc, f = self.faker()
        if name is None:
            name = f"{f.first_name()} {f.last_name()}"
        parts = [p.strip(".,").lower() for p in name.split() if p.isalpha()] or ["mail"]
        parts = [p.encode("ascii", "ignore").decode() or "user" for p in parts]
        first, last = parts[0], parts[-1]
        local = self.pick([f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}", f"{first}_{last}", f"{first}", f"{last}.{first}", f"{first}.{last}{self.digits(2)}", f"{first}{self.digits(4)}"])
        domain = self.pick([f.free_email_domain(), f.domain_name(), "gmail.com", "outlook.com", "proton.me", "company.com", "example.org", "mail.co.uk", "yahoo.co.jp", "web.de"])
        return f"{local}@{domain}"

    def url(self) -> str:
        loc, f = self.faker()
        d = f.domain_name()
        path = self.pick(["", "", "/", f"/{f.word()}", f"/{f.word()}/{f.word()}", f"/{f.word()}?id={self.digits(3)}", f"/docs/{f.word()}#section", f"/{f.word()}/{self.digits(4)}/{f.word()}.html"])
        return self.pick([f"https://{d}{path}", f"https://www.{d}{path}", f"http://{d}{path}", f"www.{d}{path}", f"{d}{path}", f"https://github.com/{f.user_name()}/{f.word()}", f"https://{d}:{self.pick(['8080', '3000', '443'])}{path}"])

    def phone(self) -> str:
        d = self.digits
        forms = [
            f"+1 ({d(3)}) {d(3)}-{d(4)}", f"({d(3)}) {d(3)}-{d(4)}", f"{d(3)}-{d(3)}-{d(4)}", f"{d(3)}.{d(3)}.{d(4)}",
            f"+1-{d(3)}-{d(3)}-{d(4)}", f"1 {d(3)} {d(3)} {d(4)}", f"+1 {d(3)} {d(3)} {d(4)}", f"({d(3)}) {d(3)} {d(4)}",
            f"+44 20 {d(4)} {d(4)}", f"020 {d(4)} {d(4)}", f"+44 7{d(3)} {d(6)}", f"07{d(3)} {d(6)}", f"+44 (0)20 {d(4)} {d(4)}",
            f"+49 30 {d(6)}", f"+49 {d(3)} {d(7)}", f"030 {d(6)}", f"0{d(3)} {d(5)}{d(2)}", f"+49 (0) {d(2)} {d(3)} {d(4)}",
            f"+33 1 {d(2)} {d(2)} {d(2)} {d(2)}", f"0{d(1)} {d(2)} {d(2)} {d(2)} {d(2)}", f"+33 6 {d(2)} {d(2)} {d(2)} {d(2)}",
            f"+91 {d(5)} {d(5)}", f"+91-{d(5)}-{d(5)}", f"0{d(2)}-{d(8)}", f"+91 {d(2)} {d(4)} {d(4)}",
            f"+61 2 {d(4)} {d(4)}", f"04{d(2)} {d(3)} {d(3)}", f"(02) {d(4)} {d(4)}",
            f"+81 3-{d(4)}-{d(4)}", f"03-{d(4)}-{d(4)}", f"090-{d(4)}-{d(4)}",
            f"+55 11 9{d(4)}-{d(4)}", f"(11) 9{d(4)}-{d(4)}",
            f"+7 ({d(3)}) {d(3)}-{d(2)}-{d(2)}", f"8 ({d(3)}) {d(3)}-{d(2)}-{d(2)}",
            f"+86 {d(3)} {d(4)} {d(4)}", f"+82 10-{d(4)}-{d(4)}", f"+31 6 {d(8)}", f"+31 20 {d(3)} {d(4)}",
            f"+34 {d(3)} {d(3)} {d(3)}", f"+39 0{d(1)} {d(4)} {d(4)}", f"+41 {d(2)} {d(3)} {d(2)} {d(2)}",
            f"+46 8 {d(3)} {d(3)} {d(2)}", f"+48 {d(3)} {d(3)} {d(3)}", f"+90 {d(3)} {d(3)} {d(2)} {d(2)}",
            f"+52 55 {d(4)} {d(4)}", f"+27 {d(2)} {d(3)} {d(4)}", f"+65 {d(4)} {d(4)}", f"+971 {d(2)} {d(3)} {d(4)}",
            f"{d(3)} {d(3)} {d(4)}", f"{d(10)}", f"+{d(2)} {d(9)}", f"{d(4)} {d(6)}", f"({d(3)}){d(3)}-{d(4)}",
        ]
        p = self.pick(forms)
        if self.chance(0.08):
            p += self.pick([" ext. ", " x", " ext ", ", ext. "]) + self.digits(self.pick([2, 3, 4]))
        return p

    def money(self) -> str:
        r = self.rng
        whole = self.pick([r.randint(1, 99), r.randint(100, 999), r.randint(1000, 99999), r.randint(100000, 9999999), r.randint(1, 20)])
        cents = r.randint(0, 99)
        def grp(sep: str) -> str:
            s = f"{whole:,}"
            return s.replace(",", sep)
        def indian() -> str:
            s = str(whole)
            if len(s) <= 3:
                return s
            head, tail = s[:-3], s[-3:]
            parts = []
            while len(head) > 2:
                parts.insert(0, head[-2:])
                head = head[:-2]
            if head:
                parts.insert(0, head)
            return ",".join(parts) + "," + tail
        c2 = f"{cents:02d}"
        forms = [
            f"${grp(',')}.{c2}", f"${grp(',')}", f"US${grp(',')}.{c2}", f"USD {grp(',')}.{c2}", f"{grp(',')}.{c2} USD", f"$ {grp(',')}.{c2}",
            f"€{grp('.')},{c2}", f"{grp('.')},{c2} €", f"{grp(' ')},{c2} €", f"EUR {grp(',')}.{c2}", f"{grp('.')},{c2} EUR", f"€ {grp('.')},{c2}", f"{grp(',')}.{c2}€",
            f"£{grp(',')}.{c2}", f"£{grp(',')}", f"GBP {grp(',')}.{c2}",
            f"¥{grp(',')}", f"JPY {grp(',')}", f"{grp(',')}円", f"CN¥{grp(',')}.{c2}", f"RMB {grp(',')}",
            f"₹{indian()}.{c2}", f"Rs. {indian()}", f"INR {indian()}", f"₹ {indian()}",
            f"CHF {grp(chr(39))}.{c2}", f"Fr. {grp(chr(39))}.{c2}", f"SFr. {grp(',')}.{c2}",
            f"R$ {grp('.')},{c2}", f"R${grp('.')},{c2}", f"BRL {grp(',')}.{c2}",
            f"{grp('.')},{c2} kr", f"{grp(' ')},{c2} kr", f"kr {grp('.')},{c2}", f"SEK {grp(' ')}", f"{grp(' ')} kr", f"DKK {grp('.')},{c2}", f"NOK {grp(' ')},{c2}",
            f"{grp(' ')},{c2} zł", f"PLN {grp(' ')},{c2}", f"{grp(' ')} Kč", f"CZK {grp(' ')}", f"{grp(' ')} Ft", f"HUF {grp(' ')}",
            f"₽{grp(' ')}", f"{grp(' ')} ₽", f"{grp(' ')},{c2} руб.", f"RUB {grp(' ')}",
            f"₩{grp(',')}", f"KRW {grp(',')}", f"{grp(',')}원",
            f"₪{grp(',')}.{c2}", f"ILS {grp(',')}", f"₺{grp('.')},{c2}", f"{grp('.')},{c2} TL", f"TRY {grp(',')}",
            f"A${grp(',')}.{c2}", f"AU$ {grp(',')}", f"AUD {grp(',')}.{c2}", f"C${grp(',')}.{c2}", f"CA$ {grp(',')}.{c2}", f"CAD {grp(',')}",
            f"NZ${grp(',')}.{c2}", f"HK${grp(',')}", f"S${grp(',')}.{c2}", f"SGD {grp(',')}", f"MX${grp(',')}.{c2}", f"MXN {grp(',')}",
            f"R {grp(' ')},{c2}", f"ZAR {grp(',')}.{c2}", f"₱{grp(',')}.{c2}", f"PHP {grp(',')}", f"฿{grp(',')}", f"THB {grp(',')}",
            f"₫{grp('.')}", f"{grp('.')} ₫", f"VND {grp(',')}", f"Rp {grp('.')}", f"IDR {grp(',')}", f"AED {grp(',')}.{c2}", f"SAR {grp(',')}",
            f"${whole}k", f"${whole}K", f"€{whole}k", f"${whole}M", f"${r.randint(1, 9)}.{r.randint(1, 9)}M", f"£{whole}m", f"{whole} million dollars", f"{whole} bucks", f"{whole} euros", f"{whole} dollars", f"{whole} USD", f"{whole} EUR",
            f"${whole}.{c2}", f"€{whole},{c2}", f"{whole},{c2}€", f"{whole}.{c2} USD", f"US$ {whole}", f"{whole}€", f"{whole}$", f"{whole} £",
        ]
        m = self.pick(forms)
        if self.chance(0.05):
            m = "-" + m
        return m

    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
    MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def date(self) -> str:
        r = self.rng
        y = r.randint(1998, 2031)
        mi = r.randint(0, 11)
        d = r.randint(1, 28)
        mon = self.MONTHS[mi]
        mon3 = mon[:3]
        hh = r.randint(0, 23)
        mm = self.pick(["00", "15", "30", "45", f"{r.randint(0, 59):02d}"])
        h12 = (hh % 12) or 12
        ampm = "am" if hh < 12 else "pm"
        tz = self.pick(["Z", "+00:00", "+02:00", "-05:00", "+05:30", "-08:00"])
        day = self.pick(self.DAYS)
        suffix = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        forms = [
            f"{y}-{mi + 1:02d}-{d:02d}", f"{y}-{mi + 1:02d}-{d:02d}T{hh:02d}:{mm}:00{tz}", f"{y}-{mi + 1:02d}-{d:02d} {hh:02d}:{mm}",
            f"{mon} {d}, {y}", f"{mon} {d}", f"{mon3} {d}, {y}", f"{mon3} {d}", f"{mon3}. {d}, {y}", f"{d} {mon} {y}", f"{d} {mon3} {y}", f"{d} {mon3}", f"{d}{suffix} {mon}",
            f"{mon} {d}{suffix}", f"{mon} {d}{suffix}, {y}", f"the {d}{suffix} of {mon}", f"{d}.{mi + 1}.{y}", f"{d:02d}.{mi + 1:02d}.{y}", f"{d:02d}/{mi + 1:02d}/{y}", f"{mi + 1:02d}/{d:02d}/{y}",
            f"{mi + 1}/{d}/{y}", f"{mi + 1}/{d}/{str(y)[2:]}", f"{d}/{mi + 1}/{str(y)[2:]}", f"{y}/{mi + 1:02d}/{d:02d}", f"{y}.{mi + 1:02d}.{d:02d}", f"{mon} {y}", f"{mon3} {y}", f"Q{r.randint(1, 4)} {y}", f"FY{y}", f"{y}",
            f"{day}", f"{day}, {mon} {d}", f"{day} {d} {mon}", f"{day}, {d} {mon3} {y}", f"{day[:3]}, {mon3} {d}", f"{day[:3]} {mon3} {d} {y}",
            f"{h12}:{mm} {ampm.upper()}", f"{h12}:{mm}{ampm}", f"{h12}{ampm}", f"{hh:02d}:{mm}", f"{hh:02d}:{mm}:00", f"{h12} o'clock", f"{h12}:{mm} {ampm} {self.pick(['EST', 'PST', 'CET', 'UTC', 'IST', 'GMT'])}",
            f"{day} at {h12}{ampm}", f"{day} {h12}:{mm}", f"{mon3} {d} at {h12}:{mm} {ampm}", f"{mon} {d}, {y} at {h12}:{mm} {ampm}", f"{d} {mon3} {y}, {hh:02d}:{mm}", f"{day}, {mon} {d}, {y}",
            f"tomorrow", f"today", f"yesterday", f"tonight", f"next {day}", f"last {day}", f"this {day}", f"next week", f"next month", f"end of {mon}", f"early {mon}", f"mid-{mon}", f"late {y}", f"in {r.randint(2, 10)} days", f"{r.randint(2, 6)} weeks ago", f"tomorrow at {h12}{ampm}", f"tomorrow morning", f"{day} morning", f"{day} afternoon", f"{day} evening", f"noon", f"midnight", f"end of day", f"EOD {day}", f"COB {day[:3]}", f"next {day} at {h12}", f"{h12}:{mm} tomorrow", f"the week of {mon3} {d}",
            f"{d} {self.MONTHS_FR[mi]} {y}", f"{d}. {self.MONTHS_DE[mi]} {y}", f"{d} de {self.MONTHS_ES[mi]} de {y}", f"{d} {self.MONTHS_FR[mi]}", f"{d}. {self.MONTHS_DE[mi]}", f"{d} de {self.MONTHS_ES[mi]}", f"{self.MONTHS_DE[mi]} {y}",
            f"{y}{mi + 1:02d}{d:02d}", f"{mon3} {d}-{d + 2}", f"{mon} {d}–{d + 3}, {y}", f"{y}-{y + 1}",
            f"{day[:3]}, {d:02d} {mon3} {y} {hh:02d}:{mm}:00 +0000",
        ]
        return self.pick(forms)

    def address_lines(self) -> list[str]:
        """Postal address as lines, in one of 15+ national layouts."""
        loc, f = self.faker(english_bias=0.5)
        r = self.rng
        country = COUNTRY.get(loc, "")
        street = f.street_address()
        city = f.city()
        try:
            postcode = f.postcode()
        except Exception:  # noqa: BLE001
            postcode = self.digits(5)
        state = ""
        for attr in ("state_abbr", "state", "administrative_unit", "province", "region", "county", "prefecture"):
            if hasattr(f, attr):
                try:
                    state = getattr(f, attr)()
                    break
                except Exception:  # noqa: BLE001
                    continue
        unit = self.pick(["", "", "", "", f"Apt {r.randint(1, 40)}", f"Suite {r.randint(100, 999)}", f"Unit {r.randint(1, 30)}", f"Floor {r.randint(1, 20)}", f"#{r.randint(1, 500)}", f"{r.randint(1, 9)}. OG", f"Flat {r.randint(1, 20)}"])
        roll = r.random()
        if loc in ("en_US", "en_CA") or (loc.startswith("en") and roll < 0.2):
            lines = [street] + ([unit] if unit else []) + [f"{city}, {state} {postcode}"]
        elif loc in ("en_GB", "en_IE"):
            lines = [street] + ([unit] if unit and r.random() < 0.5 else []) + [city, postcode] if r.random() < 0.6 else [street, f"{city} {postcode}"]
        elif loc in ("en_AU", "en_NZ"):
            lines = [street, f"{city} {state} {postcode}"]
        elif loc == "en_IN":
            lines = [street, f"{city} - {postcode}", state] if r.random() < 0.5 else [street, f"{city}, {state} {postcode}"]
        elif loc.startswith("de") or loc in ("nl_NL", "cs_CZ", "pl_PL", "da_DK", "no_NO", "sv_SE", "fi_FI", "hu_HU", "ro_RO", "it_IT", "es_ES", "fr_FR", "pt_PT", "tr_TR", "ru_RU"):
            lines = [street, f"{postcode} {city}"]
        elif loc in ("ja_JP", "ko_KR", "zh_CN"):
            raw = f.address()
            lines = [ln for ln in raw.replace("\r", "").split("\n") if ln.strip()] or [street, city]
        else:
            raw = f.address()
            lines = [ln for ln in raw.replace("\r", "").split("\n") if ln.strip()] or [street, city]
        if country and self.chance(0.35):
            lines.append(country if self.chance(0.8) else country.upper())
        if self.chance(0.08):
            lines[0] = self.pick(["PO Box ", "P.O. Box ", "Postfach ", "BP "]) + self.digits(r.randint(2, 5))
        if self.chance(0.05):
            lines = [ln.upper() for ln in lines]
        return [ln.strip() for ln in lines if ln.strip()]

    def address(self, multiline: bool | None = None) -> str:
        lines = self.address_lines()
        if multiline is None:
            multiline = self.chance(0.5)
        sep = "\n" if multiline else self.pick([", ", ", ", " , ", " ", " · ", " | "])
        return sep.join(lines)

    # -- sentence templates ----------------------------------------------------------------
    FILLERS = [
        "Thanks for getting back to me so quickly.", "Let me know if that works for you.", "I hope this helps.",
        "Please find the details below.", "That sounds good to me.", "We should probably talk about this in person.",
        "The meeting has been moved to the larger room.", "I'll send the updated version later today.",
        "It rained all weekend, so we stayed in and watched films.", "The garden needs watering twice a week in summer.",
        "Honestly, the second draft reads much better than the first.", "Can you double check the numbers before we ship?",
        "The train was late again, which is becoming a habit.", "Our team grew by three people this quarter.",
        "The recipe calls for two eggs and a pinch of salt.", "She said the results were promising but not conclusive.",
        "Nobody expected the demo to go that smoothly.", "Remember to bring your badge to the front desk.",
        "The new library opens next to the old post office.", "We walked along the river until it got dark.",
        "Most of the feedback was about the onboarding flow.", "There is still a lot of work to do on the docs.",
        "The kids built a fort out of every cushion in the house.", "His talk covered caching, retries, and backpressure.",
        "I finally finished the book you recommended.", "The coffee machine on the third floor is broken again.",
        "We are considering a slightly different approach.", "It was colder than the forecast suggested.",
        "The update fixes two crashes and a memory leak.", "Everyone agreed the logo needed more contrast.",
        "Sorry for the late reply, it has been a hectic week.", "Attached is the summary from yesterday's session.",
        "Could you forward this to the rest of the group?", "The vote passed with a narrow majority.",
        "Migration is scheduled during the maintenance window.", "The paint took two days to dry properly.",
        "I think we are overcomplicating this.", "Lunch is on me if the build passes.",
        "The museum was quieter than expected on a Sunday.", "Quick question about the invoice from last month.",
        "The dog refuses to go out when it is windy.", "Their support team was surprisingly helpful.",
        "Not sure this is the right thread, but here goes.", "The bug only reproduces on the second login.",
        "We had a great time, thanks again for hosting.", "The lease runs for another eighteen months.",
        "You can ignore the earlier message.", "Both options have trade-offs worth discussing.",
        "The forecast says clear skies for the rest of the week.", "I'll be offline for the next couple of hours.",
        "This is the third time the printer has jammed today.", "Great work on the presentation!",
        "ok", "sounds good", "will do", "thanks!", "see you then", "noted, thanks", "lol", "yes please", "no worries",
        "on it", "got it, thank you", "perfect", "hmm, not sure about that", "fine by me", "🙂", "👍 thanks", "brb",
    ]
    TEMPLATES = [
        "Hi {person}, thanks for the call on {date}.", "{company} will invoice {money} by {date}.",
        "Reach me at {phone} or {email}.", "Call {person} at {phone} if anything comes up.",
        "The invoice from {company} for {money} is due {date}.", "Please send the package to {address}.",
        "{person} from {company} will join us {date}.", "We paid {money} for the tickets.",
        "Meeting with {person} {date} at the {company} office.", "Our new address is {address}.",
        "I spoke with {person} ({company}) about the {money} quote.", "Deadline is {date}, budget {money}.",
        "You can reach {person} on {phone}.", "The order shipped {date} to {address}.",
        "{company} raised {money} last year.", "Interview with {person} scheduled for {date}.",
        "{person} <{email}> asked about the refund of {money}.", "Send your CV to {email} before {date}.",
        "Tickets are {money} each; doors open {date}.", "Contact: {person}, {phone}.",
        "{company} is based at {address}.", "Rent is {money} per month starting {date}.",
        "Text {phone} to confirm.", "As discussed with {person}, the fee is {money}.",
        "We met {person} at the conference in {city} on {date}.", "{person} moved to {city} in {date}.",
        "Deliver to {person}, {address}.", "The {company} contract ({money}) renews {date}.",
        "Visit {url} for details.", "Reply to {email} by {date}.", "See {url} or call {phone}.",
        "{person} and {person2} are presenting {date}.", "Between {company} and {company2}, I prefer the latter.",
        "Shipping to {address} costs {money}.", "{person} owes me {money} since {date}.",
        "{company}'s office ({address}) is closed {date}.", "Booked for {date}, confirmation sent to {email}.",
        "Transfer {money} to {company} before {date}.", "Reminder: dentist {date}.", "Payday is {date}!",
        "That laptop was {money}?!", "Happy birthday {person}!", "Congrats to {person} on the new role at {company}.",
        "cc {person}, {person2}", "Total: {money}. Paid {date}.", "{person} — {phone} — {email}",
        "The hotel is at {address}, about ten minutes from the venue.", "Rendez-vous le {date} avec {person}.",
        "Rechnung über {money} von {company}, fällig am {date}.", "Reunión con {person} el {date}.",
        "{person} zahlt {money}.", "Le colis a été livré à {address}.", "Der Termin ist am {date}.",
        "Nous avons payé {money} à {company}.", "Ligue para {phone} até {date}.", "Envía el pago de {money} a {company}.",
    ]

    def city(self) -> str:
        loc, f = self.faker()
        return f.city()

    def sentence(self, b: Builder, mixed: bool) -> None:
        if not mixed or self.chance(0.25):
            b.add(self.pick(self.FILLERS))
            return
        tpl = self.pick(self.TEMPLATES)
        i = 0
        while i < len(tpl):
            if tpl[i] == "{":
                j = tpl.index("}", i)
                slot = tpl[i + 1 : j]
                if slot.startswith("person"):
                    b.add(self.person(), "person")
                elif slot.startswith("company"):
                    b.add(self.company(), "company")
                elif slot == "address":
                    b.add(self.address(multiline=False), "address")
                elif slot == "date":
                    b.add(self.date(), "date")
                elif slot == "money":
                    b.add(self.money(), "money")
                elif slot == "phone":
                    b.add(self.phone(), "phone")
                elif slot == "email":
                    b.add(self.email())
                elif slot == "url":
                    b.add(self.url())
                elif slot == "city":
                    b.add(self.city())
                i = j + 1
            else:
                j = tpl.find("{", i)
                if j < 0:
                    j = len(tpl)
                b.add(tpl[i:j])
                i = j

    # -- kinds ---------------------------------------------------------------------
    def gen_prose(self) -> Example:
        b = Builder()
        n = self.pick([1, 1, 1, 2, 2, 3, 4, 5])
        mixed = self.chance(0.6)
        for i in range(n):
            if i:
                b.add(self.pick([" ", " ", "  ", "\n", "\n\n"]))
            self.sentence(b, mixed)
        if self.chance(0.05):
            b.add(self.pick([" #" + self.pick(["update", "news", "reminder", "wip"]), " @" + self.pick(["team", "alex", "sam_k", "everyone"])]))
        return b.build("prose")

    def gen_address(self) -> Example:
        b = Builder()
        b.add(self.address(), "address")
        return b.build("address")

    TITLES = ["CEO", "CTO", "Software Engineer", "Senior Developer", "Product Manager", "Account Executive", "Sales Director", "Founder", "Managing Director", "Head of Design", "Marketing Lead", "Partner", "Consultant", "Project Manager", "Office Manager", "Data Scientist", "Attorney", "Realtor", "Nurse Practitioner", "Freelance Photographer", "Geschäftsführer", "Directeur commercial", "Ingeniero de software", "Staff Engineer", "VP Engineering", "Customer Success", "Recruiter", "Editor"]

    def gen_contact(self) -> Example:
        b = Builder()
        name = self.person()
        company = self.company()
        style = self.rng.random()
        email = self.email(name)
        if style < 0.28:  # signature block
            if self.chance(0.5):
                b.add(self.pick(["-- \n", "--\n", "Best,\n", "Regards,\n", "Cheers,\n", "Thanks,\n", "Kind regards,\n", "Mit freundlichen Grüßen\n", "Cordialement,\n", "Saludos,\n", "Best regards,\n\n"]))
            b.add(name, "person").add("\n")
            if self.chance(0.7):
                b.add(self.pick(self.TITLES))
                b.add(self.pick([" | ", ", ", " at ", " @ ", " · ", " - ", "\n"]))
                b.add(company, "company").add("\n")
            elif self.chance(0.5):
                b.add(company, "company").add("\n")
            if self.chance(0.75):
                b.add(self.pick(["", "T: ", "Tel: ", "M: ", "Mobile: ", "P: ", "Phone: ", "Tel. ", "Cell: ", "📞 ", "☎ "]))
                b.add(self.phone(), "phone").add("\n")
            if self.chance(0.6):
                b.add(self.pick(["", "E: ", "Email: ", "✉ ", "e-mail: "])).add(email).add("\n")
            if self.chance(0.4):
                b.add(self.pick(["", "W: ", "Web: ", "🌐 "])).add(self.url()).add("\n")
            if self.chance(0.4):
                b.add(self.address(multiline=self.chance(0.5)), "address").add("\n")
        elif style < 0.5:  # contact card / labelled
            b.add(self.pick(["Name: ", "Contact: ", "Full name: ", "", "Nom : ", "Name: "])).add(name, "person").add("\n")
            fields = []
            if self.chance(0.7):
                fields.append(("company", self.pick(["Company: ", "Org: ", "Employer: ", "Firma: ", "Organisation: "]), company))
            if self.chance(0.5):
                fields.append(("title", self.pick(["Title: ", "Role: ", "Position: ", "Job title: "]), self.pick(self.TITLES)))
            if self.chance(0.85):
                fields.append(("phone", self.pick(["Phone: ", "Tel: ", "Mobile: ", "Cell: ", "Téléphone : ", "Telefon: ", "Phone (work): ", "Fax: "]), self.phone()))
            if self.chance(0.75):
                fields.append(("email", self.pick(["Email: ", "E-mail: ", "Mail: ", "E: "]), email))
            if self.chance(0.45):
                fields.append(("address", self.pick(["Address: ", "Addr: ", "Adresse : ", "Location: ", "Office: "]), self.address(multiline=self.chance(0.3))))
            if self.chance(0.3):
                fields.append(("url", self.pick(["Website: ", "LinkedIn: ", "URL: ", "Web: "]), self.url()))
            if self.chance(0.2):
                fields.append(("date", self.pick(["Birthday: ", "DOB: ", "Joined: ", "Last contact: "]), self.date()))
            self.rng.shuffle(fields)
            for kind, label, value in fields:
                b.add(label)
                b.add(value, kind if kind in SPAN_KINDS else None)
                b.add("\n")
        elif style < 0.7:  # one-liners
            form = self.rng.random()
            if form < 0.2:
                b.add(name, "person").add(" <").add(email).add(">")
            elif form < 0.4:
                if form < 0.3:
                    b.add(name, "person").add(" (").add(company, "company").add(") ").add(self.phone(), "phone")
                else:
                    b.add(name, "person").add(self.pick([" - ", " – ", ", "])).add(company, "company").add(self.pick([", ", " | ", " - "])).add(self.phone(), "phone")
            elif form < 0.55:
                b.add(name, "person").add(self.pick([", ", " | ", " — ", " · "])).add(self.pick(self.TITLES)).add(self.pick([", ", " | ", " at ", " @ "])).add(company, "company")
            elif form < 0.7:
                b.add(name, "person").add(self.pick([": ", " ", " - ", ", "])).add(self.phone(), "phone")
            elif form < 0.85:
                b.add(name, "person").add(self.pick([" | ", ", ", " "])).add(email).add(self.pick([" | ", ", ", " "])).add(self.phone(), "phone")
            else:
                b.add(company, "company").add(self.pick([" | ", ", ", "\n", " — "])).add(self.phone(), "phone").add(self.pick([" | ", ", ", "\n", " — "])).add(self.address(multiline=False), "address")
        elif style < 0.85:  # business card style: name, title, company, address, phone
            b.add(name, "person").add("\n")
            if self.chance(0.7):
                b.add(self.pick(self.TITLES)).add("\n")
            b.add(company, "company").add("\n")
            if self.chance(0.7):
                b.add(self.address(multiline=self.chance(0.6)), "address").add("\n")
            if self.chance(0.8):
                b.add(self.phone(), "phone").add("\n")
            if self.chance(0.6):
                b.add(email).add("\n")
        elif style < 0.93:  # bare name (with optional title)
            b.add(name, "person")
            if self.chance(0.3):
                b.add(self.pick([", ", " - ", " — "])).add(self.pick(self.TITLES))
        else:  # company + address / phone
            b.add(company, "company").add("\n")
            if self.chance(0.7):
                b.add(self.address(multiline=self.chance(0.6)), "address").add("\n")
            if self.chance(0.6):
                b.add(self.pick(["", "Tel: ", "Phone: "])).add(self.phone(), "phone").add("\n")
            if self.chance(0.5):
                b.add(self.url())
        return b.build("contact")

    ITEMS = ["milk", "eggs", "bread", "butter", "coffee beans", "olive oil", "tomatoes", "bananas", "rice", "chicken thighs", "dish soap", "paper towels", "batteries", "toothpaste", "avocados", "oat milk", "spinach", "cheddar", "yoghurt", "pasta", "garlic", "lemons", "cereal", "honey", "flour",
             "fix login bug", "write tests for the parser", "review PR", "update dependencies", "book flights", "call the landlord", "renew passport", "send invoice", "prepare slides", "clean the garage", "schedule dentist", "reply to recruiter", "buy birthday gift", "deploy to staging", "rotate API keys", "draft the announcement", "water the plants", "pick up dry cleaning", "backup laptop", "order new chairs",
             "Paris", "Tokyo", "Lisbon", "Berlin", "Toronto", "Mumbai", "Seoul", "Cape Town", "Oslo", "Mexico City",
             "React", "Vue", "Svelte", "Rust", "Python", "TypeScript", "PostgreSQL", "Redis", "Kubernetes", "Terraform",
             "Inception", "Parasite", "Arrival", "Heat", "Spirited Away", "Amélie", "Whiplash", "Dune", "Her", "Coco"]

    def gen_list(self) -> Example:
        b = Builder()
        r = self.rng
        n = r.randint(2, 9)
        style = r.random()
        if style < 0.12:  # comma list
            items = r.sample(self.ITEMS, n)
            sep = self.pick([", ", ", ", "; ", " / ", " • ", " | "])
            b.add(sep.join(items))
            return b.build("list")
        bullet_kind = self.pick(["- ", "* ", "• ", "· ", "– ", "— ", "+ ", "", "", "[ ] ", "[x] ", "☐ ", "✅ ", "→ ", ">> ", "num", "num)", "alpha", "roman"])
        entity = self.pick([None, None, None, "person", "money", "date", "phone", "company", "mixed"])
        if self.chance(0.25):
            b.add(self.pick(["Shopping:", "TODO", "To do:", "Agenda", "Attendees:", "Groceries", "Packing list", "Next steps:", "Ideas", "Cities to visit", "Things to bring", "Notes from standup:", "Options:"])).add("\n")
        for i in range(n):
            if bullet_kind == "num":
                b.add(f"{i + 1}. ")
            elif bullet_kind == "num)":
                b.add(f"{i + 1}) ")
            elif bullet_kind == "alpha":
                b.add(f"{string.ascii_lowercase[i]}) ")
            elif bullet_kind == "roman":
                b.add(f"{['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix'][i]}. ")
            else:
                b.add(bullet_kind)
            kind = entity if entity != "mixed" else self.pick(SPAN_KINDS + [None, None])
            if kind == "person":
                b.add(self.person(), "person")
                if self.chance(0.4):
                    b.add(self.pick([" - ", ": ", " (", " — "])).add(self.pick(self.TITLES + self.ITEMS[25:45]))
                    if ")" in b.parts[-2]:
                        b.add(")")
            elif kind == "money":
                b.add(self.pick(self.ITEMS[:25])).add(self.pick([" - ", ": ", " ", " … ", "  "])).add(self.money(), "money")
            elif kind == "date":
                b.add(self.pick(self.ITEMS[25:45])).add(self.pick([" - ", ": ", " by ", " due ", " (", " @ "])).add(self.date(), "date")
                if "(" in b.parts[-2]:
                    b.add(")")
            elif kind == "phone":
                b.add(self.person(), "person").add(self.pick([": ", " - ", " "])).add(self.phone(), "phone")
            elif kind == "company":
                b.add(self.company(), "company")
                if self.chance(0.3):
                    b.add(self.pick([" - ", ": "])).add(self.money(), "money")
            elif kind == "address":
                b.add(self.address(multiline=False), "address")
            else:
                item = self.pick(self.ITEMS)
                if self.chance(0.15):
                    item = item.capitalize()
                b.add(item)
                if self.chance(0.15):
                    b.add(f" x{r.randint(2, 6)}")
            if i < n - 1:
                b.add("\n")
        return b.build("list")

    def ident(self) -> str:
        w = self.pick(["user", "item", "count", "total", "index", "result", "data", "name", "value", "config", "buffer", "node", "list", "price", "order", "token", "query", "key", "path", "event", "handler", "client", "server", "response", "request", "cache", "size", "limit", "offset", "row", "col", "id", "flag", "status", "message", "error", "logger", "temp", "max", "min", "sum", "avg"])
        style = self.rng.random()
        if style < 0.3:
            return w
        if style < 0.6:
            return w + self.pick(["s", "Count", "List", "Map", "Id", "Name", "Total", "Value", "_id", "_name", "_count", "Index", "Cache"])
        if style < 0.8:
            return w + "_" + self.pick(["a", "b", "x", "tmp", "out", "in", "new", "old", "next", "prev"])
        return self.pick(["i", "j", "k", "n", "x", "y", "acc", "el", "fn", "cb", "ctx", "res", "req", "err", "db", "tx"])

    def gen_code(self) -> Example:
        r = self.rng
        i1, i2, i3, i4 = self.ident(), self.ident(), self.ident(), self.ident()
        n1, n2 = r.randint(0, 100), r.randint(1, 1000)
        s1 = self.pick(["hello", "done", "error", "ok", "user not found", "loading…", "Retrying", "invalid input", "TODO", "%s items"])
        cls = self.pick(["User", "Order", "Cache", "Parser", "Client", "Node", "Config", "Service", "Queue", "Handler"])
        lang = self.pick(["js", "ts", "py", "go", "rs", "java", "c", "rb", "sh", "sql", "css", "php", "swift", "kt", "cs", "yaml"])
        snippets = {
            "js": [
                f"function {i1}({i2}) {{\n  return {i2}.map(x => x * {n1});\n}}",
                f"const {i1} = require('{i2}');\nconst {i3} = {i1}.{i4}({n1});\nconsole.log({i3});",
                f"export const {i1} = async ({i2}) => {{\n  const {i3} = await fetch(`/api/{i2}/${{{i2}.id}}`);\n  if (!{i3}.ok) throw new Error('{s1}');\n  return {i3}.json();\n}};",
                f"for (let i = 0; i < {i1}.length; i++) {{\n  if ({i1}[i] > {n1}) {i2}.push({i1}[i]);\n}}",
                f"{i1}.addEventListener('click', () => {{\n  {i2}.classList.toggle('{i3}');\n}});",
                f"const {{ {i1}, {i2} }} = {i3};\nreturn {i1} ?? {i2};",
                f"module.exports = {{ {i1}, {i2}: {n1}, {i3}: '{s1}' }};",
                f"setTimeout(() => {i1}({n1}), {n2});",
                f"if ({i1} === undefined) {{\n  throw new TypeError('{s1}');\n}}",
                f"const {i1} = {i2}.filter(Boolean).reduce((acc, x) => acc + x, 0);",
            ],
            "ts": [
                f"interface {cls} {{\n  id: number;\n  {i1}: string;\n  {i2}?: {cls}[];\n}}",
                f"export function {i1}<T>({i2}: T[]): T | undefined {{\n  return {i2}[{n1}];\n}}",
                f"type {cls} = {{ {i1}: string; {i2}: number }};\nconst {i3}: {cls} = {{ {i1}: '{s1}', {i2}: {n1} }};",
                f"const {i1} = new Map<string, {cls}>();\n{i1}.set('{i2}', {i3});",
                f"export async function {i1}({i2}: string): Promise<void> {{\n  await {i3}.{i4}({i2});\n}}",
                f"enum {cls} {{ {i1.capitalize()} = 'x', {i2.capitalize()} = 'y' }}",
                f"import {{ {i1} }} from './{i2}.js';\nimport type {{ {cls} }} from './types.js';",
            ],
            "py": [
                f"def {i1}({i2}, {i3}=None):\n    if {i3} is None:\n        {i3} = []\n    return [x * {n1} for x in {i2}]",
                f"import os\nimport sys\n\n{i1} = os.environ.get('{i2.upper()}', '{s1}')\nprint({i1})",
                f"class {cls}:\n    def __init__(self, {i1}):\n        self.{i1} = {i1}\n\n    def {i2}(self):\n        return self.{i1} + {n1}",
                f"for {i1} in range({n1}):\n    {i2}.append({i1} ** 2)",
                f"with open('{i1}.txt') as f:\n    {i2} = f.read().splitlines()",
                f"try:\n    {i1} = int({i2})\nexcept ValueError:\n    {i1} = {n1}",
                f"@dataclass\nclass {cls}:\n    {i1}: int = {n1}\n    {i2}: str = '{s1}'",
                f"{i1} = {{k: v for k, v in {i2}.items() if v > {n1}}}",
                f"if __name__ == '__main__':\n    {i1}()",
                f"import numpy as np\n{i1} = np.zeros(({n1}, {n2}))\n{i1}[:, 0] = 1",
            ],
            "go": [
                f"func {i1}({i2} []int) int {{\n\ttotal := 0\n\tfor _, v := range {i2} {{\n\t\ttotal += v\n\t}}\n\treturn total\n}}",
                f"package main\n\nimport \"fmt\"\n\nfunc main() {{\n\tfmt.Println(\"{s1}\")\n}}",
                f"if err != nil {{\n\treturn nil, fmt.Errorf(\"{i1}: %w\", err)\n}}",
                f"type {cls} struct {{\n\tID   int    `json:\"id\"`\n\t{i1.capitalize()} string `json:\"{i1}\"`\n}}",
                f"{i1} := make(map[string]int)\n{i1}[\"{i2}\"] = {n1}",
            ],
            "rs": [
                f"fn {i1}({i2}: &[i32]) -> i32 {{\n    {i2}.iter().sum()\n}}",
                f"let {i1}: Vec<u32> = (0..{n1}).collect();\nprintln!(\"{{:?}}\", {i1});",
                f"#[derive(Debug, Clone)]\nstruct {cls} {{\n    {i1}: String,\n    {i2}: u64,\n}}",
                f"match {i1} {{\n    Some(v) => v,\n    None => return Err(\"{s1}\".into()),\n}}",
                f"impl {cls} {{\n    pub fn new({i1}: &str) -> Self {{\n        Self {{ {i1}: {i1}.to_string(), {i2}: {n1} }}\n    }}\n}}",
            ],
            "java": [
                f"public class {cls} {{\n    private int {i1};\n    public int get{i1.capitalize()}() {{\n        return {i1};\n    }}\n}}",
                f"List<String> {i1} = new ArrayList<>();\n{i1}.add(\"{s1}\");",
                f"for (int i = 0; i < {n1}; i++) {{\n    System.out.println({i1}[i]);\n}}",
                f"@Override\npublic String toString() {{\n    return \"{cls}(\" + {i1} + \")\";\n}}",
            ],
            "c": [
                f"#include <stdio.h>\n\nint main(void) {{\n    printf(\"{s1}\\n\");\n    return 0;\n}}",
                f"int {i1}(int *{i2}, size_t n) {{\n    int s = 0;\n    for (size_t i = 0; i < n; i++) s += {i2}[i];\n    return s;\n}}",
                f"struct {cls} {{\n    int {i1};\n    char {i2}[{n2}];\n}};",
                f"#define {i1.upper()} {n2}\nstatic int {i2}[{i1.upper()}];",
                f"std::vector<int> {i1}({n1});\nstd::sort({i1}.begin(), {i1}.end());",
            ],
            "rb": [
                f"def {i1}({i2})\n  {i2}.map {{ |x| x * {n1} }}\nend",
                f"class {cls}\n  attr_reader :{i1}\n  def initialize({i1})\n    @{i1} = {i1}\n  end\nend",
                f"{i1}.each do |{i2}|\n  puts {i2}\nend",
            ],
            "sh": [
                f"#!/bin/bash\nset -euo pipefail\n{i1}=$(ls | wc -l)\necho \"$${i1}\"",
                f"for f in *.log; do\n  gzip \"$f\"\ndone",
                f"curl -s https://api.{i1}.com/v1/{i2} | jq '.{i3}'",
                f"export {i1.upper()}={n1}\nnpm run {i2} -- --{i3}",
                f"git checkout -b {i1}/{i2}\ngit push -u origin {i1}/{i2}",
                f"docker run -p {n2}:{n2} -e {i1.upper()}={i2} {i3}:latest",
                f"sudo apt-get install -y {i1} {i2}",
                f"ssh {i1}@{i2}.example.com 'tail -f /var/log/{i3}.log'",
            ],
            "sql": [
                f"SELECT {i1}, COUNT(*) FROM {i2} WHERE {i3} > {n1} GROUP BY {i1} ORDER BY 2 DESC;",
                f"INSERT INTO {i1} ({i2}, {i3}) VALUES ('{s1}', {n1});",
                f"CREATE TABLE {i1} (\n  id SERIAL PRIMARY KEY,\n  {i2} TEXT NOT NULL,\n  {i3} INTEGER DEFAULT {n1}\n);",
                f"UPDATE {i1} SET {i2} = {n1} WHERE id = {n2};",
                f"select * from {i1} u join {i2} o on o.{i3} = u.id limit {n1};",
            ],
            "css": [
                f".{i1} {{\n  display: flex;\n  gap: {n1}px;\n  color: #{self.digits(3)};\n}}",
                f"@media (max-width: {n2}px) {{\n  .{i1} {{ display: none; }}\n}}",
                f"#{i1} > .{i2}:hover {{\n  background: rgba(0, 0, 0, 0.{n1:02d});\n  transition: all 0.2s ease;\n}}",
            ],
            "php": [
                f"<?php\n${i1} = ${i2}->{i3}({n1});\necho ${i1};",
                f"function {i1}(array ${i2}): int {{\n    return count(${i2}) * {n1};\n}}",
            ],
            "swift": [
                f"let {i1} = {i2}.map {{ $0 * {n1} }}",
                f"struct {cls}: Codable {{\n    let {i1}: String\n    var {i2}: Int = {n1}\n}}",
                f"guard let {i1} = {i2} else {{ return }}",
            ],
            "kt": [
                f"data class {cls}(val {i1}: String, val {i2}: Int = {n1})",
                f"fun {i1}({i2}: List<Int>): Int = {i2}.sumOf {{ it * {n1} }}",
                f"val {i1} = {i2}?.let {{ it.{i3} }} ?: return",
            ],
            "cs": [
                f"public class {cls}\n{{\n    public int {i1.capitalize()} {{ get; set; }} = {n1};\n}}",
                f"var {i1} = {i2}.Where(x => x.{i3} > {n1}).ToList();",
                f"Console.WriteLine($\"{s1}: {{{i1}}}\");",
            ],
            "yaml": [
                f"{i1}:\n  {i2}: {n1}\n  {i3}: \"{s1}\"\n  {i4}:\n    - a\n    - b",
                f"version: '3'\nservices:\n  {i1}:\n    image: {i2}:latest\n    ports:\n      - \"{n2}:{n2}\"",
                f"[{i1}]\n{i2} = {n1}\n{i3} = \"{s1}\"",
                f"{i1.upper()}={n1}\n{i2.upper()}={s1}\n{i3.upper()}=true",
            ],
        }
        b = Builder()
        parts = [self.pick(snippets[lang])]
        if self.chance(0.3):
            parts.append(self.pick(snippets[lang]))
        code = "\n\n".join(parts) if self.chance(0.5) else "\n".join(parts)
        if self.chance(0.2):
            comment = {"py": "# ", "sh": "# ", "rb": "# ", "yaml": "# ", "sql": "-- "}.get(lang, "// ")
            b.add(comment + self.pick(["TODO: clean this up", "FIXME", "see issue #" + self.digits(3), "temporary workaround", "handles the edge case", "do not remove"]) + "\n")
        b.add(code)
        if self.chance(0.15):
            b.add("\n" + self.pick(["  ", "\t", "    "]))
        return b.build("code")

    def gen_markdown(self) -> Example:
        b = Builder()
        r = self.rng
        blocks = r.randint(1, 4)
        used_heading = False
        for i in range(blocks):
            if i:
                b.add("\n\n" if self.chance(0.8) else "\n")
            kind = self.pick(["heading", "heading", "para", "para", "bullets", "link", "code", "quote", "table", "checks", "bold"])
            if kind == "heading" and not used_heading:
                used_heading = True
                b.add(self.pick(["# ", "## ", "### ", "#### "]) + self.pick(["Overview", "Installation", "Usage", "Notes", "Changelog", "Meeting notes", "Getting started", "API", "Roadmap", "Summary", "Weekly update", "Release 2.1", "FAQ"]))
                if self.chance(0.5):
                    b.add("\n\n")
                    self.sentence(b, self.chance(0.5))
            elif kind == "para" or kind == "heading":
                self.sentence(b, self.chance(0.5))
                if self.chance(0.5):
                    b.add(" ")
                    self.sentence(b, self.chance(0.4))
            elif kind == "bullets":
                n = r.randint(2, 5)
                bullet = self.pick(["- ", "* ", "+ ", "num"])
                for j in range(n):
                    b.add(f"{j + 1}. " if bullet == "num" else bullet)
                    if self.chance(0.3):
                        b.add("**" + self.pick(self.ITEMS[25:45]) + "**: ")
                        self.sentence(b, self.chance(0.5))
                    elif self.chance(0.3):
                        b.add("`" + self.ident() + "`" + self.pick([" - ", ": "]) + self.pick(self.FILLERS[:20]))
                    else:
                        b.add(self.pick(self.ITEMS))
                    if j < n - 1:
                        b.add("\n")
            elif kind == "link":
                b.add(self.pick(["See ", "Docs: ", "More at ", "Read the ", ""])).add("[" + self.pick(["docs", "the guide", "this page", "README", "issue tracker", "here"]) + "](" + self.url() + ")")
                if self.chance(0.5):
                    b.add(self.pick([" for details.", ".", " and the ", ""]))
                    if b.parts[-1] == " and the ":
                        b.add("![screenshot](" + self.url() + ")")
            elif kind == "code":
                b.add("```" + self.pick(["", "js", "ts", "python", "bash", "sh", "json"]) + "\n")
                b.add(self.pick(["npm install " + self.ident(), "pip install " + self.ident(), "const " + self.ident() + " = " + str(r.randint(1, 99)) + ";", "print(" + self.ident() + ")", "git pull --rebase", "{ \"" + self.ident() + "\": " + str(r.randint(1, 99)) + " }", "curl -X POST " + self.url()]))
                b.add("\n```")
            elif kind == "quote":
                b.add("> ")
                self.sentence(b, self.chance(0.5))
            elif kind == "table":
                c1, c2 = self.pick(["Name", "Item", "Task", "Package"]), self.pick(["Status", "Price", "Owner", "Size"])
                b.add(f"| {c1} | {c2} |\n|---|---|\n")
                n = r.randint(2, 4)
                for j in range(n):
                    b.add("| ")
                    if c1 == "Name":
                        b.add(self.person(), "person")
                    else:
                        b.add(self.pick(self.ITEMS))
                    b.add(" | ")
                    if c2 == "Price":
                        b.add(self.money(), "money")
                    elif c2 == "Owner":
                        b.add(self.person(), "person")
                    else:
                        b.add(self.pick(["done", "open", "12 KB", "blocked", "in review", "3 MB"]))
                    b.add(" |")
                    if j < n - 1:
                        b.add("\n")
            elif kind == "checks":
                n = r.randint(2, 5)
                for j in range(n):
                    b.add(self.pick(["- [ ] ", "- [x] ", "* [ ] "]))
                    b.add(self.pick(self.ITEMS[25:45]))
                    if self.chance(0.3):
                        b.add(" (due ").add(self.date(), "date").add(")")
                    if j < n - 1:
                        b.add("\n")
            else:
                b.add(self.pick(["**Note:** ", "*Important:* ", "**TL;DR** ", "_Update:_ ", "~~old~~ new: "]))
                self.sentence(b, self.chance(0.5))
        return b.build("markdown")

    def gen_structured(self) -> Example:
        """JSON / CSV / TSV / HTML / logs / key=value with entities inside; kind is masked."""
        b = Builder()
        r = self.rng
        style = self.pick(["json", "json", "csv", "csv", "tsv", "html", "log", "kv", "email_header"])
        name, company, money, date, phone = self.person(), self.company(), self.money(), self.date(), self.phone()
        if style == "json":
            b.add("{\n  \"" + self.pick(["name", "customer", "contact", "fullName"]) + "\": \"").add(name, "person").add("\",\n")
            if self.chance(0.6):
                b.add("  \"" + self.pick(["company", "org", "employer"]) + "\": \"").add(company, "company").add("\",\n")
            if self.chance(0.6):
                b.add("  \"email\": \"" + self.email(name) + "\",\n")
            if self.chance(0.5):
                b.add("  \"phone\": \"").add(phone, "phone").add("\",\n")
            if self.chance(0.5):
                b.add("  \"" + self.pick(["amount", "total", "price"]) + "\": \"").add(money, "money").add("\",\n")
            if self.chance(0.5):
                b.add("  \"" + self.pick(["date", "created", "due"]) + "\": \"").add(date, "date").add("\",\n")
            if self.chance(0.4):
                b.add("  \"address\": \"").add(self.address(multiline=False), "address").add("\",\n")
            b.add("  \"id\": " + str(r.randint(1, 9999)) + "\n}")
        elif style in ("csv", "tsv"):
            sep = "\t" if style == "tsv" else self.pick([",", ",", ";"])
            cols = self.pick([["name", "email", "phone"], ["Name", "Company", "Amount"], ["date", "description", "amount"], ["id", "name", "city"], ["Contact", "Phone", "Address"]])
            b.add(sep.join(cols)).add("\n")
            for j in range(r.randint(2, 5)):
                cells = []
                for c in cols:
                    cl = c.lower()
                    if cl in ("name", "contact"):
                        b.add(self.person(), "person")
                    elif cl == "company":
                        b.add(self.company(), "company")
                    elif cl == "email":
                        b.add(self.email())
                    elif cl == "phone":
                        b.add(self.phone(), "phone")
                    elif cl == "amount":
                        b.add(self.money(), "money")
                    elif cl == "date":
                        b.add(self.date(), "date")
                    elif cl == "address":
                        a = self.address(multiline=False)
                        if sep in a:
                            b.add('"').add(a, "address").add('"')
                        else:
                            b.add(a, "address")
                    elif cl == "city":
                        b.add(self.city())
                    elif cl == "id":
                        b.add(str(r.randint(1, 999)))
                    else:
                        b.add(self.pick(self.ITEMS))
                    cells.append(c)
                    if c != cols[-1]:
                        b.add(sep)
                if j < 4:
                    b.add("\n")
        elif style == "html":
            b.add("<div class=\"" + self.pick(["card", "contact", "vcard", "row"]) + "\">\n  <h3>").add(name, "person").add("</h3>\n")
            if self.chance(0.6):
                b.add("  <p class=\"org\">").add(company, "company").add("</p>\n")
            if self.chance(0.6):
                b.add("  <a href=\"mailto:" + self.email(name) + "\">Email</a>\n")
            if self.chance(0.5):
                b.add("  <span class=\"tel\">").add(phone, "phone").add("</span>\n")
            if self.chance(0.4):
                b.add("  <address>").add(self.address(multiline=False), "address").add("</address>\n")
            if self.chance(0.4):
                b.add("  <time>").add(date, "date").add("</time>\n")
            b.add("</div>")
        elif style == "log":
            for j in range(r.randint(2, 5)):
                b.add(self.date(), "date").add(self.pick([" [INFO] ", " INFO ", " [ERROR] ", " WARN ", " DEBUG "]))
                b.add(self.pick(["user ", "payment from ", "order for ", "login by ", "invoice to ", "call from "]))
                kind = self.pick(["person", "company", "money", "phone"])
                b.add({"person": self.person, "company": self.company, "money": self.money, "phone": self.phone}[kind](), kind)
                b.add(self.pick([" ok", " failed", " retrying", " (" + str(r.randint(1, 999)) + "ms)", ""]))
                if j < 4:
                    b.add("\n")
        elif style == "kv":
            b.add(self.pick(["customer", "CUSTOMER", "owner", "billed_to"]) + "=").add(name, "person").add("\n")
            b.add(self.pick(["vendor", "COMPANY", "supplier"]) + "=").add(company, "company").add("\n")
            b.add("amount=").add(money, "money").add("\n")
            b.add("due=").add(date, "date").add("\n")
            b.add("tel=").add(phone, "phone")
        else:
            b.add("From: ").add(name, "person").add(" <" + self.email(name) + ">\n")
            b.add("To: ").add(self.person(), "person").add(" <" + self.email() + ">\n")
            b.add("Date: ").add(date, "date").add("\n")
            b.add("Subject: " + self.pick(["Invoice ", "Re: meeting ", "Payment of ", "Fwd: quote from "]))
            if b.parts[-1].startswith(("Payment", "Fwd")):
                b.add(money if "Payment" in b.parts[-1] else company, "money" if "Payment" in b.parts[-1] else "company")
            else:
                b.add(self.digits(4))
        return b.build(None)

    def gen_single(self) -> Example:
        """A bare entity as the whole paste (kind derived by the decoder)."""
        b = Builder()
        kind = self.pick(["person", "company", "date", "money", "phone", "address"])
        value = {"person": self.person, "company": self.company, "date": self.date, "money": self.money, "phone": self.phone, "address": lambda: self.address()}[kind]()
        b.add(value, kind)
        label = {"person": "contact", "company": "contact", "address": "address"}.get(kind)
        return b.build(label)

    # -- noise ----------------------------------------------------------------------
    def noise(self, ex: Example) -> Example:
        text = ex.text
        spans = ex.spans
        roll = self.rng.random()
        if roll < 0.04:
            text = text.lower()
        elif roll < 0.06:
            text = text.upper()
        if self.chance(0.1):
            pad = self.pick([" ", "\n", "  ", "\n\n", "\t"])
            spans = [(s + _utf16(pad), e + _utf16(pad), k) for s, e, k in spans]
            text = pad + text
        if self.chance(0.1):
            text = text + self.pick([" ", "\n", "\n\n", "  "])
        return Example(text, ex.kind, spans)

    def example(self) -> Example:
        roll = self.rng.random()
        if roll < 0.15:
            ex = self.gen_address()
        elif roll < 0.33:
            ex = self.gen_contact()
        elif roll < 0.55:
            ex = self.gen_prose()
        elif roll < 0.67:
            ex = self.gen_list()
        elif roll < 0.79:
            ex = self.gen_code()
        elif roll < 0.89:
            ex = self.gen_markdown()
        elif roll < 0.96:
            ex = self.gen_structured()
        else:
            ex = self.gen_single()
        return self.noise(ex)


def generate(n: int, seed: int, offline: bool = False) -> list[Example]:
    g = Gen(seed, offline=offline)
    return [g.example() for _ in range(n)]


def to_json(ex: Example) -> dict:
    return {"text": ex.text, "kind": ex.kind, "spans": [{"span": [s, e], "kind": k} for s, e, k in ex.spans]}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for ex in generate(n, seed=7):
        print(json.dumps(to_json(ex), ensure_ascii=False))
        print("-" * 60)
        print(ex.text)
        for s, e, k in ex.spans:
            print(f"   [{k}] {ex.text.encode('utf-16-le')[s * 2 : e * 2].decode('utf-16-le')!r}")
        print("=" * 60)


if __name__ == "__main__":
    main()
