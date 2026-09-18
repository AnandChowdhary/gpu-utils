"""Synthetic email generator and dataset builder for gpu-email.

An email is built as a list of Line(text, kind, spans) and then joined with a newline
style. Line kinds follow features.LINE_KINDS; spans are contact fields (features.FIELDS)
inside the author's own signature, as [start, end) character offsets within the line.
Token labels are derived from the lines after tokenization, so the generator never has
to know about tokens.

`uv run python -m gpu_email.data` writes the cached train/heldout tensors and
downloads the real evaluation fixtures (talon, email_reply_parser) into data/cache.
"""

from __future__ import annotations

import json
import random
import string
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpu_email import vocab as V
from gpu_email.features import BIO_LABELS, FIELDS, LINE_KINDS, NUM_SLOTS, featurize_tokens
from gpu_utils_training.features import CLASS_NEWLINE, CLASS_SPACE, tokenize

KIND = {k: i for i, k in enumerate(LINE_KINDS)}
BIO = {k: i for i, k in enumerate(BIO_LABELS)}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class Line:
    text: str
    kind: int
    spans: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class Person:
    locale: str
    first: str
    last: str
    full: str  # display name in the email
    email: str
    title: str
    company: str
    phone: str
    phone2: str
    url: str
    address: list[str]
    native: str | None = None  # JA/ZH script name


@dataclass
class Example:
    text: str
    line_kinds: list[int]  # per line, -1 for blank
    spans: list[tuple[int, int, str]]  # absolute char offsets
    contact: dict[str, object]
    reply: str
    meta: dict[str, object]


# ---------------------------------------------------------------------------------
# Random helpers


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def p(self, prob: float) -> bool:
        return self.r.random() < prob

    def pick(self, xs):
        return self.r.choice(xs)

    def maybe(self, prob: float, xs):
        return self.r.choice(xs) if self.r.random() < prob else None

    def digits(self, n: int) -> str:
        return "".join(self.r.choice("0123456789") for _ in range(n))

    def alnum(self, n: int) -> str:
        return "".join(self.r.choice(string.ascii_lowercase + string.digits) for _ in range(n))

    # --- names, companies, contacts ------------------------------------------------

    def locale(self) -> str:
        return self.r.choices(
            ["en", "fr", "de", "es", "it", "nl", "pt", "sv", "pl", "ja", "zh", "ar"],
            weights=[52, 8, 9, 8, 3, 3, 3, 2, 2, 4, 4, 2],
        )[0]

    def phone(self, locale: str) -> str:
        r = self.r
        d = self.digits
        fmt = {
            "en": [
                "({a}) {b}-{c}", "{a}-{b}-{c}", "{a}.{b}.{c}", "+1 {a} {b} {c}", "+1 ({a}) {b}-{c}",
                "+1-{a}-{b}-{c}", "{a} {b} {c}", "1-800-{b}-{c}", "+44 20 {d4} {d4}", "+44 7{d3} {d6}",
                "020 {d4} {d4}", "07{d3} {d6}", "+91 {d5} {d5}", "+91-{d5}-{d5}", "0{d4} {d6}",
                "+61 4{d2} {d3} {d3}", "+1.{a}.{b}.{c}", "{a}{b}{c}", "+353 1 {d3} {d4}", "+1 {a}-{b}-{c} ext. {d3}",
                "+1 {a} {b} {c} x{d3}", "({a}) {b}-{c} ext {d2}", "+65 {d4} {d4}", "+971 4 {d3} {d4}",
            ],
            "fr": ["0{d1} {d2} {d2} {d2} {d2}", "+33 {d1} {d2} {d2} {d2} {d2}", "+33 (0){d1} {d2} {d2} {d2} {d2}", "0{d1}.{d2}.{d2}.{d2}.{d2}", "+41 {d2} {d3} {d2} {d2}", "+32 {d1} {d3} {d2} {d2}"],
            "de": ["+49 {d2} {d4} {d4}", "+49 {d3} {d5}{d2}", "0{d2} {d4}-{d3}", "+49 (0) {d2} {d3} {d3}", "0{d3}/{d5}", "+49 1{d2} {d7}", "+43 1 {d3} {d4}", "+41 44 {d3} {d2} {d2}", "0{d3} {d6}"],
            "es": ["+34 {d3} {d3} {d3}", "{d3} {d2} {d2} {d2}", "6{d2} {d3} {d3}", "+34 6{d2} {d2} {d2} {d2}", "+52 55 {d4} {d4}", "+54 11 {d4} {d4}", "+57 1 {d3} {d4}"],
            "it": ["+39 0{d2} {d6}", "+39 3{d2} {d7}", "3{d2} {d3} {d4}", "0{d1} {d7}"],
            "nl": ["+31 6 {d8}", "06-{d8}", "+31 20 {d3} {d4}", "020 {d3} {d4}"],
            "pt": ["+351 9{d2} {d3} {d3}", "+55 11 9{d4}-{d4}", "9{d2} {d3} {d3}", "(11) 9{d4}-{d4}"],
            "sv": ["+46 70 {d3} {d2} {d2}", "070-{d3} {d2} {d2}", "+46 8 {d3} {d3} {d2}"],
            "pl": ["+48 {d3} {d3} {d3}", "{d3} {d3} {d3}", "+48 22 {d3} {d2} {d2}"],
            "ja": ["03-{d4}-{d4}", "+81 3-{d4}-{d4}", "090-{d4}-{d4}", "+81 90 {d4} {d4}", "06-{d4}-{d4}"],
            "zh": ["+86 1{d2} {d4} {d4}", "1{d2}-{d4}-{d4}", "+86 10 {d4} {d4}", "010-{d8}", "1{d10}"],
            "ar": ["+971 50 {d3} {d4}", "+966 5{d1} {d3} {d4}", "+20 10 {d4} {d4}", "050 {d3} {d4}"],
        }[locale]
        t = r.choice(fmt)
        a = r.choice(["212", "415", "646", "312", "617", "206", "303", "512", "917", "310", "702", "404", "555"])
        return t.format(a=a, b=d(3), c=d(4), d1=d(1), d2=d(2), d3=d(3), d4=d(4), d5=d(5), d6=d(6), d7=d(7), d8=d(8), d10=d(10))

    def company(self) -> str:
        r = self.r
        stem = r.choice(V.COMPANY_STEMS)
        kind = r.choice(V.COMPANY_KINDS)
        x = r.random()
        if x < 0.25:
            return f"{stem} {kind}"
        if x < 0.5:
            return f"{stem} {kind} {r.choice(V.COMPANY_SUFFIXES)}"
        if x < 0.65:
            return f"{stem} {r.choice(V.COMPANY_SUFFIXES)}"
        if x < 0.75:
            return stem
        if x < 0.85:
            return r.choice(V.COMPANY_EXTRA).format(stem=stem, kind=kind)
        if x < 0.92:
            return f"{stem}{kind}".replace(" ", "")
        return f"{stem} {kind}".upper()

    def domain_for(self, company: str) -> str:
        base = "".join(c for c in company.lower() if c.isalnum())[:14] or "example"
        return f"{base}.{self.pick(V.TLDS)}"

    def address(self, locale: str) -> list[str]:
        r = self.r
        num = str(r.randint(1, 9999))
        street = f"{num} {r.choice(V.STREET_NAMES)} {r.choice(V.STREET_TYPES)}"
        if locale in ("en", "ar") or r.random() < 0.2:
            x = r.random()
            if x < 0.45:
                city, st = r.choice(V.US_CITIES)
                z = self.digits(5)
                suite = r.choice(["", "", f", Suite {r.randint(100, 999)}", f", Floor {r.randint(2, 40)}", f" #{r.randint(1, 999)}", f", Unit {r.randint(1, 99)}"])
                if r.random() < 0.5:
                    return [f"{street}{suite}, {city}, {st} {z}"]
                return [f"{street}{suite}", f"{city}, {st} {z}"]
            if x < 0.65:
                city = r.choice(V.UK_CITIES)
                pc = r.choice(V.UK_POSTCODES)
                if r.random() < 0.5:
                    return [f"{street}, {city} {pc}"]
                return [f"{street}", f"{city}", pc] if r.random() < 0.3 else [f"{street}", f"{city} {pc}"]
            if x < 0.8:
                city, state, pin = r.choice(V.IN_CITIES)
                st2 = r.choice(V.IN_STREETS)
                return [f"{r.randint(1, 500)}, {st2}", f"{city}, {state} {pin}"] if r.random() < 0.5 else [f"{st2}, {city} - {pin}"]
            if x < 0.9:
                city, st, pc = r.choice(V.AU_CITIES)
                return [f"Level {r.randint(1, 30)}, {street}", f"{city} {st} {pc}"] if r.random() < 0.5 else [f"{street}, {city} {st} {pc}"]
            city, st, pc = r.choice(V.CA_CITIES)
            return [f"{street}, {city}, {st} {pc}"]
        if locale == "de":
            st = f"{r.choice(V.DE_STREETS)} {r.randint(1, 200)}"
            city = f"{self.digits(5)} {r.choice(V.DE_CITIES)}"
            return [st, city] if r.random() < 0.6 else [f"{st}, {city}"] if r.random() < 0.5 else [f"{st} · {city}"]
        if locale == "fr":
            st = f"{r.randint(1, 200)} {r.choice(V.FR_STREETS)}"
            city = f"{self.digits(5)} {r.choice(V.FR_CITIES)}"
            return [st, city] if r.random() < 0.6 else [f"{st}, {city}"]
        if locale in ("es", "pt", "it"):
            st = f"{r.choice(V.ES_STREETS)} {r.randint(1, 200)}"
            city = f"{self.digits(5)} {r.choice(V.ES_CITIES)}"
            return [st, city] if r.random() < 0.6 else [f"{st}, {city}"]
        if locale == "ja":
            return [r.choice(V.JA_ADDRESSES)]
        if locale == "zh":
            return [r.choice(V.ZH_ADDRESSES)]
        st = f"{r.choice(V.STREET_NAMES)}gatan {r.randint(1, 99)}"
        return [st, f"{self.digits(3)} {self.digits(2)} {r.choice(['Stockholm', 'Göteborg', 'Amsterdam', 'Warszawa', 'Kraków', 'Malmö'])}"]

    def person(self, locale: str | None = None) -> Person:
        r = self.r
        locale = locale or self.locale()
        pool = locale if locale in V.FIRST_NAMES else "en"
        first = r.choice(V.FIRST_NAMES[pool])
        last = r.choice(V.LAST_NAMES[pool])
        native = None
        if locale == "ja":
            native = r.choice(V.JA_NAMES)
        elif locale == "zh":
            native = r.choice(V.ZH_NAMES)
        company = self.company()
        domain = self.domain_for(company) if r.random() < 0.7 else r.choice(V.EMAIL_DOMAINS_PERSONAL)
        fl = first.lower()
        ll = last.lower().replace(" ", "").replace("'", "")
        for a, b in (("é", "e"), ("è", "e"), ("ü", "ue"), ("ö", "oe"), ("ä", "ae"), ("ß", "ss"), ("ñ", "n"), ("í", "i"), ("á", "a"), ("ó", "o"), ("ç", "c"), ("ł", "l"), ("ś", "s"), ("ń", "n"), ("ą", "a"), ("ę", "e"), ("ż", "z"), ("ź", "z")):
            fl = fl.replace(a, b)
            ll = ll.replace(a, b)
        local = r.choice([f"{fl}.{ll}", f"{fl}{ll}", f"{fl[0]}{ll}", f"{fl}", f"{fl}_{ll}", f"{ll}", f"{fl}.{ll}{r.randint(1, 99)}", f"{fl[0]}.{ll}", f"{fl}-{ll}", f"{ll}.{fl[0]}"])
        email = f"{local}@{domain}"
        if r.random() < 0.15:
            email = email.upper() if r.random() < 0.3 else email.title()
        full = f"{first} {last}"
        x = r.random()
        if x < 0.06:
            full = f"{first} {last[0]}."
        elif x < 0.1:
            full = f"{first[0]}. {last}"
        elif x < 0.14:
            full = f"{first} {r.choice(string.ascii_uppercase)}. {last}"
        elif x < 0.17:
            full = full.upper()
        elif x < 0.2:
            full = f"{last}, {first}"
        elif x < 0.24:
            full = r.choice([f"Dr. {first} {last}", f"{first} {last}, PhD", f"{first} {last}, MD", f"Prof. {first} {last}", f"{first} {last}, CPA", f"{first} {last} Jr.", f"{first} {last}, Esq.", f"Mr. {first} {last}", f"Ms. {first} {last}", f"Dr {first} {last}"])
        elif x < 0.27:
            full = f"{first} {r.choice(V.FIRST_NAMES[pool])} {last}"
        # only forms the exact URL regex in the decoder can recover (http(s):// or www.)
        url_style = r.choice(["https://www.{d}", "https://{d}", "http://www.{d}", "www.{d}", "https://{d}/", "https://www.linkedin.com/in/{fl}{ll}", "www.linkedin.com/in/{fl}-{ll}", "https://{d}/team/{fl}", "https://calendly.com/{fl}-{ll}/30min", "https://twitter.com/{fl}{ll}", "https://github.com/{fl}{ll}", "www.{d}/{fl}"])
        url = url_style.format(d=self.domain_for(company), fl=fl, ll=ll)
        return Person(
            locale=locale, first=first, last=last, full=full, email=email,
            title=r.choice(V.TITLES), company=company, phone=self.phone(locale), phone2=self.phone(locale),
            url=url, address=self.address(locale), native=native,
        )

    # --- dates -------------------------------------------------------------------

    def date_parts(self):
        r = self.r
        y = r.randint(2009, 2026)
        m = r.randint(1, 12)
        d = r.randint(1, 28)
        wd = r.randint(0, 6)
        h = r.randint(0, 23)
        mi = r.randint(0, 59)
        return y, m, d, wd, h, mi

    def date_en(self, style: str | None = None) -> str:
        r = self.r
        y, m, d, wd, h, mi = self.date_parts()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_l = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        wds = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wds_l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        t12 = f"{h12}:{mi:02d} {ampm}"
        t12b = f"{h12}:{mi:02d} {ampm.lower()}"
        t24 = f"{h:02d}:{mi:02d}"
        style = style or r.choice(["gmail", "gmail2", "apple", "apple2", "tb", "tb2", "outlook", "outlook2", "yahoo", "iso", "short", "long", "uk", "uk2", "gh"])
        return {
            "gmail": f"{wds[wd]}, {months[m - 1]} {d}, {y} at {t12}",
            "gmail2": f"{wds[wd]}, {d} {months[m - 1]} {y} at {t24}",
            "apple": f"{months[m - 1]} {d}, {y}, at {t12}",
            "apple2": f"{d} {months[m - 1]} {y}, at {t24}",
            "tb": f"{y}-{m:02d}-{d:02d} {t24}",
            "tb2": f"{m:02d}/{d:02d}/{y % 100:02d} {t12}",
            "outlook": f"{wds_l[wd]}, {months_l[m - 1]} {d:02d}, {y} {t12}",
            "outlook2": f"{d:02d} {months_l[m - 1]} {y} {t24}",
            "yahoo": f"{wds[wd]}, {m}/{d}/{y % 100:02d}",
            "iso": f"{y}-{m:02d}-{d:02d}T{t24}",
            "short": f"{m}/{d}/{y}",
            "long": f"{wds_l[wd]}, {months_l[m - 1]} {d}, {y}",
            "uk": f"{d} {months_l[m - 1]} {y} {t24}",
            "uk2": f"{wds[wd]}, {d} {months[m - 1]} {y} at {t12b}",
            "gh": f"{months[m - 1]} {d}, {y} {t12}",
        }[style]

    def date_local(self, locale: str) -> str:
        r = self.r
        y, m, d, wd, h, mi = self.date_parts()
        t = f"{h:02d}:{mi:02d}"
        if locale == "fr":
            months = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
            wds = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
            return r.choice([f"{d} {months[m - 1]} {y} à {t}", f"{wds[wd]} {d} {months[m - 1]} {y} à {t}", f"{d}/{m:02d}/{y} {h:02d}:{mi:02d}", f"{d} {months[m - 1]} {y}"])
        if locale == "de":
            months = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
            wds = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
            return r.choice([f"{d:02d}.{m:02d}.{y} um {t}", f"{d}. {months[m - 1]} {y} um {t}", f"{wds[wd]}, {d}. {months[m - 1]} {y} {t}", f"{d:02d}.{m:02d}.{y} {t}", f"{d:02d}.{m:02d}.{y}"])
        if locale == "es":
            months = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"]
            wds = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
            return r.choice([f"{d} {months[m - 1]} {y}, a las {t}", f"{wds[wd]}, {d} {months[m - 1]} {y} a las {t}", f"{d} de {months[m - 1]} de {y}", f"{d}/{m}/{y} {t}", f"{wds[wd]}., {d} {months[m - 1]}. {y} {t}"])
        if locale == "it":
            months = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
            wds = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]
            return r.choice([f"{wds[wd]} {d} {months[m - 1]} {y} alle ore {t}", f"{d} {months[m - 1]} {y}, alle ore {t}", f"{d}/{m:02d}/{y} {t}"])
        if locale == "nl":
            months = ["jan.", "feb.", "mrt.", "apr.", "mei", "jun.", "jul.", "aug.", "sep.", "okt.", "nov.", "dec."]
            wds = ["ma", "di", "wo", "do", "vr", "za", "zo"]
            return r.choice([f"{wds[wd]} {d} {months[m - 1]} {y} om {t}", f"{d}-{m:02d}-{y} {t}", f"{d} {months[m - 1]} {y} {t}"])
        if locale == "pt":
            months = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
            wds = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
            return r.choice([f"{wds[wd]}., {d} de {months[m - 1]}. de {y} às {t}", f"{d}/{m:02d}/{y} {t}", f"{d} de {months[m - 1]} de {y} às {t}"])
        if locale == "sv":
            months = ["jan.", "feb.", "mars", "apr.", "maj", "juni", "juli", "aug.", "sep.", "okt.", "nov.", "dec."]
            wds = ["mån", "tis", "ons", "tors", "fre", "lör", "sön"]
            return r.choice([f"{wds[wd]} {d} {months[m - 1]} {y} kl. {t}", f"{y}-{m:02d}-{d:02d} {t}"])
        if locale == "pl":
            wds = ["pon.", "wt.", "śr.", "czw.", "pt.", "sob.", "niedz."]
            return r.choice([f"{wds[wd]}, {d} {['sty', 'lut', 'mar', 'kwi', 'maj', 'cze', 'lip', 'sie', 'wrz', 'paź', 'lis', 'gru'][m - 1]} {y} o {t}", f"{d}.{m:02d}.{y} {t}"])
        if locale == "ja":
            wds = ["月", "火", "水", "木", "金", "土", "日"]
            return r.choice([f"{y}年{m}月{d}日({wds[wd]}) {t}", f"{y}年{m}月{d}日 {t}", f"{y}/{m:02d}/{d:02d} {t}", f"{y}年{m}月{d}日({wds[wd]}) {h}:{mi:02d}"])
        if locale == "zh":
            wds = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            return r.choice([f"{y}年{m}月{d}日 {wds[wd]} {t}", f"{y}年{m}月{d}日 {h}:{mi:02d}", f"{y}-{m:02d}-{d:02d} {t}", f"{y}年{m}月{d}日，{t}"])
        return self.date_en()


# ---------------------------------------------------------------------------------
# Building blocks


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


class EmailGen(Gen):
    def __init__(self, seed: int):
        super().__init__(seed)

    # --- text filling ---------------------------------------------------------------

    def fill(self, template: str, me: Person, other: Person) -> str:
        r = self.r
        pool = me.locale if me.locale in V.FIRST_NAMES else "en"
        return template.format(
            first=other.first, first2=r.choice(V.FIRST_NAMES[pool]), last=other.last, full=other.full,
            title_last=f"{r.choice(['Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Mr', 'Ms', 'M.', 'Mme', 'Herr', 'Frau', 'Sr.', 'Sra.'])} {other.last}",
            last_ja=(other.native or other.last).split(" ")[0], full_ja=other.native or other.full,
            last_zh=(other.native or other.last)[0], full_zh=other.native or other.full,
            noun=r.choice(V.NOUNS), noun2=r.choice(V.NOUNS), day=r.choice(V.DAYS), day2=r.choice(V.DAYS),
            time=r.choice(V.TIMES), name=r.choice(V.FIRST_NAMES[pool]), q=r.randint(1, 4),
            url=r.choice([me.url, other.url, f"https://{self.domain_for(me.company)}/{self.alnum(6)}", f"http://wiki.{self.domain_for(other.company)}/{self.alnum(4)}.html#{self.alnum(3)}"]),
            amount=f"{r.randint(1, 99999):,}", os=r.choice(V.OSES), num=r.randint(1, 99999),
            address=", ".join(me.address), phone=r.choice([me.phone, other.phone]),
            email=r.choice([me.email, other.email]), company=r.choice([me.company, other.company]),
            list=r.choice(["riak-users", "dev", "announce", "users", "python-list", "team-infra"]),
            key=r.choice(["PROJ", "INFRA", "WEB", "APP"]),
        )

    def body_lines(self, me: Person, other: Person, kind: int, max_sentences: int = 8, width: int | None = None) -> list[Line]:
        """Paragraphs of prose in `kind`. Returns lines including blank separators."""
        r = self.r
        n_par = r.choices([1, 1, 2, 2, 3, 4, 5], weights=[20, 20, 25, 15, 10, 6, 4])[0]
        out: list[Line] = []
        if width is None:
            width = r.choice([0, 0, 0, 72, 76, 78, 80, 68, 64, 100])
        for pi in range(n_par):
            n_s = r.randint(1, max_sentences)
            pool = V.BODY_SENTENCES_EN if (me.locale == "en" or r.random() < 0.5) else V.BODY_SENTENCES_BY_LOCALE.get(me.locale, V.BODY_SENTENCES_EN)
            sents = [self.fill(r.choice(pool), me, other) for _ in range(n_s)]
            joiner = " " if r.random() < 0.75 else "\n"
            para = joiner.join(sents)
            for raw in para.split("\n"):
                if width and len(raw) > width and not raw.startswith(("    ", "$ ", "- ", "* ", "• ")):
                    for w in _wrap(raw, width):
                        out.append(Line(w, kind))
                else:
                    out.append(Line(raw, kind))
            if pi < n_par - 1:
                out.append(Line("", kind))
                if r.random() < 0.08:
                    out.append(Line("", kind))
        return out

    # --- signature ------------------------------------------------------------------

    def signature(self, me: Person, tag: bool) -> list[Line]:
        """Author signature block. When tag=True, spans for contact fields are emitted."""
        r = self.r
        lines: list[Line] = []

        def L(text: str, spans: list[tuple[int, int, str]] | None = None) -> Line:
            return Line(text, KIND["signature"], spans if (tag and spans) else [])

        def span_line(parts: list[tuple[str, str | None]], sep: str) -> Line:
            """parts: (text, field|None) joined by sep, with spans computed."""
            text = ""
            spans: list[tuple[int, int, str]] = []
            for i, (t, f) in enumerate(parts):
                if i > 0:
                    text += sep
                if f:
                    spans.append((len(text), len(text) + len(t), f))
                text += t
            return L(text, spans)

        style = r.random()
        name = me.native if (me.native and r.random() < 0.7) else me.full
        name_variants = [
            [(name, "NAME")],
            [("-", None), (name, "NAME")],
            [("- ", None), (name, "NAME")],
            [("--", None), (name, "NAME")],
            [("~ ", None), (name, "NAME")],
            [(name, "NAME"), (" ", None), (r.choice(["(he/him)", "(she/her)", "(they/them)", "(he/him/his)"]), None)],
            [(name, "NAME"), (r.choice([" | ", " - ", ", ", " / ", " – "]), None), (me.title, "TITLE")],
            [(name, "NAME"), (r.choice([" | ", " - ", ", ", " / ", " · "]), None), (me.company, "COMPANY")],
            [(name, "NAME"), (" | ", None), (me.title, "TITLE"), (" | ", None), (me.company, "COMPANY")],
            [(name, "NAME"), (", ", None), (me.title, "TITLE"), (", ", None), (me.company, "COMPANY")],
            [(name, "NAME"), (" – ", None), (me.title, "TITLE"), (" – ", None), (me.company, "COMPANY")],
            [(name, "NAME"), (" ", None), (r.choice(["(" + me.company + ")", "@" + me.company.split(" ")[0]]), None)],
            [(name, "NAME"), ("  ", None), (me.title, "TITLE")],
        ]
        nv = r.choices(name_variants, weights=[40, 4, 4, 2, 1, 3, 8, 6, 8, 5, 3, 2, 2])[0]
        name_line = span_line(nv, "")
        phone_labels = ["Tel: ", "Tel. ", "Tel.: ", "T: ", "T ", "M: ", "M ", "Mobile: ", "Mob: ", "Mob. ", "Phone: ", "Ph: ", "P: ", "Cell: ", "C: ", "Direct: ", "D: ", "Office: ", "O: ", "Fax: ", "F: ", "Tél. : ", "Tél : ", "Téléphone : ", "Portable : ", "Telefon: ", "Telefon ", "Mobil: ", "Handy: ", "Teléfono: ", "Móvil: ", "Cel: ", "Telefono: ", "Cellulare: ", "電話: ", "TEL: ", "电话：", "手机：", "☎ ", "📞 ", "📱 ", "p: ", "m: ", "t: ", "c: ", "Phone ", "Tel ", "Mobile ", "WhatsApp: ", "Call: ", "Work: ", "Home: ", "Ext. ", "toll-free: "]
        email_labels = ["", "", "", "E: ", "E ", "Email: ", "email: ", "e-mail: ", "E-mail: ", "Mail: ", "e: ", "✉ ", "📧 ", "Courriel : ", "E-Mail: ", "Correo: ", "メール: ", "邮箱："]
        url_labels = ["", "", "", "W: ", "Web: ", "web: ", "www: ", "Website: ", "Site: ", "🌐 ", "w: ", "LinkedIn: ", "Book time: ", "Calendar: ", "Twitter: ", "GitHub: ", "Web : ", "Sitio web: "]

        comps: list[Line] = []
        # title / company lines
        tc = r.random()
        if "TITLE" not in [f for _, f in nv] and tc < 0.75:
            if tc < 0.25:
                comps.append(span_line([(me.title, "TITLE")], ""))
                if r.random() < 0.8:
                    comps.append(span_line([(me.company, "COMPANY")], ""))
            elif tc < 0.45:
                comps.append(span_line([(me.title, "TITLE"), (r.choice([" | ", ", ", " - ", " at ", " @ ", " – ", " / ", " · ", ", "]), None), (me.company, "COMPANY")], ""))
            elif tc < 0.55:
                comps.append(span_line([(me.title, "TITLE"), (", ", None), (r.choice(V.DEPARTMENTS), None)], ""))
                comps.append(span_line([(me.company, "COMPANY")], ""))
            elif tc < 0.65:
                comps.append(span_line([(me.title, "TITLE")], ""))
            else:
                comps.append(span_line([(me.company, "COMPANY")], ""))
        elif "COMPANY" not in [f for _, f in nv] and r.random() < 0.5:
            comps.append(span_line([(me.company, "COMPANY")], ""))

        # contact lines
        contacts: list[Line] = []
        n_phones = r.choices([0, 1, 1, 2, 3], weights=[20, 40, 20, 15, 5])[0]
        phones = [me.phone, me.phone2, self.phone(me.locale)][:n_phones]
        if n_phones and r.random() < 0.3:
            # pipe-joined phones on one line
            parts: list[tuple[str, str | None]] = []
            for i, ph in enumerate(phones):
                if i:
                    parts.append((r.choice([" | ", " / ", "  ", " • ", ", "]), None))
                parts.append((r.choice(phone_labels), None))
                parts.append((ph, "PHONE"))
            contacts.append(span_line(parts, ""))
        else:
            for ph in phones:
                lab = r.choice(phone_labels) if r.random() < 0.8 else ""
                contacts.append(span_line([(lab, None), (ph, "PHONE")], ""))
        if r.random() < 0.65:
            contacts.append(span_line([(r.choice(email_labels), None), (me.email, "EMAIL")], ""))
        if r.random() < 0.5:
            contacts.append(span_line([(r.choice(url_labels), None), (me.url, "URL")], ""))
        if r.random() < 0.35:
            for a in me.address:
                contacts.append(span_line([(a, "ADDRESS")], ""))
        r.shuffle(contacts)
        # occasionally merge two contact lines with a pipe
        if len(contacts) >= 2 and r.random() < 0.3:
            a, b = contacts[0], contacts[1]
            sep = r.choice([" | ", " / ", " • ", "  |  ", "   "])
            merged = Line(a.text + sep + b.text, KIND["signature"], a.spans + [(s + len(a.text) + len(sep), e + len(a.text) + len(sep), f) for s, e, f in b.spans])
            contacts = [merged] + contacts[2:]

        extras: list[Line] = []
        if r.random() < 0.12:
            extras.append(L(r.choice(["\"" + r.choice(["Stay hungry, stay foolish.", "Move fast and fix things.", "Simplicity is the ultimate sophistication.", "Done is better than perfect."]) + "\"", "Pronouns: " + r.choice(["she/her", "he/him", "they/them"]), "Working hours: Mon-Fri 9-5 " + r.choice(["CET", "PST", "IST", "GMT"]), "I may send emails outside your working hours; no need to reply outside yours.", "Follow us on LinkedIn and Twitter", "Book a meeting with me", "Out of office " + r.choice(V.DAYS) + " - " + r.choice(V.DAYS), "Please excuse brevity and typos, sent on the go", "[Logo]", "[image: " + me.company + "]", "[cid:image001.png@01D9A3B2.7E1E4F30]"])))
        if r.random() < 0.06:
            extras.append(L(f"@{me.first.lower()}{me.last.lower().replace(' ', '')}"))

        # assemble
        if style < 0.6:
            lines = [name_line] + comps + contacts + extras
        elif style < 0.75:
            lines = [name_line] + comps + extras + contacts
        elif style < 0.85:
            # one/two-line dense signature
            lines = [name_line]
            if contacts:
                # rebuild as a merged line preserving spans
                text = ""
                spans: list[tuple[int, int, str]] = []
                for i, c in enumerate(contacts[:2]):
                    if i:
                        text += " | "
                    spans += [(s + len(text), e + len(text), f) for s, e, f in c.spans]
                    text += c.text
                lines.append(Line(text, KIND["signature"], spans if tag else []))
            lines += comps[:1]
        else:
            lines = comps + [name_line] + contacts + extras
        # optional separators / delimiter
        if r.random() < 0.18:
            lines.insert(0, L(r.choice(["-- ", "--", "---", "____________________", "----------------------", "________________________________", "~~~~~~~~~~~~~~~~", "=================", "***", "**********************"])))
        if r.random() < 0.08:
            lines.append(L(r.choice(["____________________", "----------------------", "-----", "————————"])))
        if r.random() < 0.08:
            lines.append(L(r.choice(V.MOBILE_SIGS)))
        # occasionally uppercase or lowercase the whole block (rare)
        if r.random() < 0.03:
            lines = [Line(ln.text.upper(), ln.kind, ln.spans) for ln in lines]
        return lines

    # --- attribution / headers -------------------------------------------------------

    def header_block(self, sender: Person, recipient: Person, kind: int, style: str = "outlook", locale: str = "en") -> list[Line]:
        r = self.r
        subj = self.fill(r.choice(V.SUBJECTS), sender, recipient)
        keys = {
            "en": ("From", "Sent", "To", "Cc", "Subject", "Date"),
            "de": ("Von", "Gesendet", "An", "Cc", "Betreff", "Datum"),
            "fr": ("De", "Envoyé", "À", "Cc", "Objet", "Date"),
            "es": ("De", "Enviado el", "Para", "CC", "Asunto", "Fecha"),
            "it": ("Da", "Inviato", "A", "Cc", "Oggetto", "Data"),
            "nl": ("Van", "Verzonden", "Aan", "CC", "Onderwerp", "Datum"),
            "pt": ("De", "Enviada em", "Para", "Cc", "Assunto", "Data"),
            "ja": ("差出人", "送信日時", "宛先", "CC", "件名", "日付"),
            "zh": ("发件人", "发送时间", "收件人", "抄送", "主题", "日期"),
        }.get(locale, ("From", "Sent", "To", "Cc", "Subject", "Date"))
        k_from, k_sent, k_to, k_cc, k_subj, k_date = keys
        colon = r.choice([": ", ": ", ":\t", ": "]) if locale not in ("ja", "zh") else r.choice([": ", "："])
        bold = r.random() < 0.15
        wrap_key = (lambda k: f"*{k}:*" + (" " if colon != "：" else "")) if bold else (lambda k: f"{k}{colon}")
        who = r.choice([f"{sender.full} <{sender.email}>", f"{sender.full} [mailto:{sender.email}]", f"{sender.email}", f"{sender.full}", f"\"{sender.full}\" <{sender.email}>", f"{sender.last}, {sender.first} <{sender.email}>", f"{sender.native or sender.full} <{sender.email}>"])
        to = r.choice([f"{recipient.full} <{recipient.email}>", f"{recipient.email}", f"{recipient.full}", f"{recipient.full}; {self.person().full}", f"{recipient.email}; {self.person().email}", f"'{recipient.full}' <{recipient.email}>"])
        date = self.date_en(r.choice(["outlook", "outlook2", "gmail", "uk", "long", "iso"])) if locale == "en" else self.date_local(locale)
        lines = [Line(f"{wrap_key(k_from)}{who}", kind)]
        if style == "forward" and r.random() < 0.5:
            lines.append(Line(f"{wrap_key(k_date)}{date}", kind))
            lines.append(Line(f"{wrap_key(k_subj)}{subj}", kind))
            lines.append(Line(f"{wrap_key(k_to)}{to}", kind))
        else:
            lines.append(Line(f"{wrap_key(k_sent if r.random() < 0.7 else k_date)}{date}", kind))
            lines.append(Line(f"{wrap_key(k_to)}{to}", kind))
            if r.random() < 0.3:
                lines.append(Line(f"{wrap_key(k_cc)}{self.person().email}", kind))
            lines.append(Line(f"{wrap_key(k_subj)}{subj}", kind))
        if r.random() < 0.15:
            lines.append(Line(f"{wrap_key(r.choice(['Importance', 'Attachments', 'Reply-To', 'Priority']))}{r.choice(['High', 'document.pdf', sender.email, 'Normal'])}", kind))
        if r.random() < 0.1:
            # wrapped To: line continuation
            lines.insert(3, Line("    " + self.person().email + ";", kind))
        return lines

    def attribution(self, sender: Person, locale: str, width: int | None = None) -> list[Line]:
        """Line(s) that introduce a quoted message (kind=attribution)."""
        r = self.r
        who = r.choice([f"{sender.full} <{sender.email}>", f"{sender.full} <{sender.email}>", f"{sender.full}", f"{sender.email}", f"<{sender.email}>", f"{sender.full}<{sender.email}>", f"\"{sender.full}\" <{sender.email}>", f"{sender.first}", f"{sender.native or sender.full} <{sender.email}>", f"{sender.full} via {r.choice(['GitHub', 'Slack', 'Jira', 'Notion'])} <{sender.email}>"])
        K = KIND["attribution"]
        if locale == "en" or (locale not in ("fr", "de", "es", "it", "nl", "pt", "sv", "pl", "ja", "zh") or r.random() < 0.2):
            t = r.choice([
                f"On {self.date_en('gmail')}, {who} wrote:", f"On {self.date_en('gmail')} {who} wrote:", f"On {self.date_en('gmail2')}, {who} wrote:",
                f"On {self.date_en('apple')}, {who} wrote:", f"On {self.date_en('apple2')}, {who} wrote:",
                f"On {self.date_en('tb')}, {who} wrote:", f"On {self.date_en('tb2')}, {who} wrote:", f"{who} wrote:", f"{who} wrote:",
                f"On {self.date_en('uk2')}, {who} wrote:", f"On {self.date_en('gh')}, {who} wrote:",
                f"--- On {self.date_en('yahoo')}, {who} wrote:", f"On {self.date_en('long')}, {who} wrote:",
                f"On {self.date_en('gmail')}, {who} wrote", f"On {self.date_en('gmail')}, {who} wrote :",
                f"{who} wrote on {self.date_en('tb')}:", f"On {self.date_en('gmail')}, {who} said:",
                f"On {self.date_en('gmail')}, {who} wrote:", f"On {self.date_en('gmail')}, {who}wrote:",
                f"Quoting {who}:", f"At {self.date_en('tb')}, {who} wrote:", f"{self.date_en('gh')}, {who}:",
                f"On {self.date_en('gmail')}, {sender.full} via {r.choice(['Notion', 'GitHub', 'Linear'])} <{sender.email}> wrote:",
                f"{who} писал(а) {self.date_en('tb')}:",
            ])
        elif locale == "fr":
            t = r.choice([f"Le {self.date_local('fr')}, {who} a écrit :", f"Le {self.date_local('fr')}, {who} a écrit:", f"Le {self.date_local('fr')}, {who} a écrit :", f"{who} a écrit :", f"Le {self.date_local('fr')} {who} a écrit :"])
        elif locale == "de":
            t = r.choice([f"Am {self.date_local('de')} schrieb {who}:", f"Am {self.date_local('de')} hat {who} geschrieben:", f"{who} schrieb am {self.date_local('de')}:", f"Am {self.date_local('de')} schrieb {who}:", f"{who} schrieb:"])
        elif locale == "es":
            t = r.choice([f"El {self.date_local('es')}, {who} escribió:", f"El {self.date_local('es')} {who} escribió:", f"{who} escribió:"])
        elif locale == "it":
            t = r.choice([f"Il {self.date_local('it')} {who} ha scritto:", f"Il giorno {self.date_local('it')} {who} ha scritto:", f"{who} ha scritto:"])
        elif locale == "nl":
            t = r.choice([f"Op {self.date_local('nl')} schreef {who}:", f"Op {self.date_local('nl')} heeft {who} het volgende geschreven:", f"{who} schreef:"])
        elif locale == "pt":
            t = r.choice([f"Em {self.date_local('pt')}, {who} escreveu:", f"{who} escreveu:", f"Em {self.date_local('pt')} {who} escreveu:"])
        elif locale == "sv":
            t = r.choice([f"Den {self.date_local('sv')} skrev {who}:", f"{who} skrev:", f"{self.date_local('sv')} skrev {who}:"])
        elif locale == "pl":
            t = r.choice([f"{self.date_local('pl')} {who} napisał(a):", f"W dniu {self.date_local('pl')}, {who} napisał:", f"{who} napisał:"])
        elif locale == "ja":
            t = r.choice([f"{self.date_local('ja')} {who}:", f"{self.date_local('ja')} {who}：", f"{who} さんは書きました:", f"{self.date_local('ja')}に {who} が書きました:", f"{self.date_local('ja')} {who}:"])
        else:  # zh
            t = r.choice([f"在 {self.date_local('zh')}，{who} 写道：", f"{who} 于{self.date_local('zh')}写道：", f"在 {self.date_local('zh')}, {who} 写道:", f"{who} 写道：", f"{self.date_local('zh')} {who}:"])
        # wrapping of long attribution lines
        if width is None:
            width = r.choice([0, 0, 0, 72, 76, 60, 70])
        if width and len(t) > width and r.random() < 0.7:
            parts = _wrap(t, width)
            # sometimes the email gets its own line (Apple Mail style)
            return [Line(p, K) for p in parts]
        return [Line(t, K)]

    # --- quoting --------------------------------------------------------------------

    def quote_prefix(self, lines: list[Line], depth: int) -> list[Line]:
        r = self.r
        style = r.choice(["> ", "> ", "> ", ">", "> "])
        blank_style = r.choice([">", "> ", ">", ""]) if style != ">" else r.choice([">", ""])
        out = []
        for ln in lines:
            if ln.text == "":
                out.append(Line(blank_style * depth if blank_style else "", KIND["quote"] if blank_style else ln.kind))
            else:
                out.append(Line(style * depth + ln.text, KIND["quote"]))
        return out

    def quoted_message(self, sender: Person, recipient: Person, depth: int, prefixed: bool, locale: str) -> list[Line]:
        """The quoted message content (kind=quote), possibly containing nested quotes."""
        r = self.r
        K = KIND["quote"]
        inner: list[Line] = []
        if r.random() < 0.6:
            inner.append(Line(self.fill(r.choice(V.GREETINGS.get(sender.locale, V.GREETINGS["en"])), sender, recipient), K))
            inner.append(Line("", K))
        inner += [Line(ln.text, K) for ln in self.body_lines(sender, recipient, K, max_sentences=5)]
        if r.random() < 0.5:
            inner.append(Line("", K))
            inner.append(Line(r.choice(V.CLOSINGS.get(sender.locale, V.CLOSINGS["en"])), K))
            if r.random() < 0.7:
                inner += [Line(ln.text, K) for ln in self.signature(sender, tag=False)]
        if r.random() < 0.15:
            inner.append(Line("", K))
            inner.append(Line(r.choice(V.MOBILE_SIGS), K))
        if r.random() < 0.15:
            inner.append(Line("", K))
            inner += [Line(t, K) for t in self.disclaimer_text(sender)]
        # nested quote inside the quoted message
        if depth < 3 and r.random() < 0.35:
            third = self.person()
            inner.append(Line("", K))
            if prefixed:
                inner += [Line(ln.text, K) for ln in self.attribution(third, locale)]
                inner += self.quoted_message(third, sender, depth + 1, True, locale)
            else:
                # unprefixed nested Outlook chain: marker + headers keep attribution kind
                inner += self.original_message_block(third, sender, locale)
                inner += self.quoted_message(third, sender, depth + 1, False, locale)
        if prefixed:
            return self.quote_prefix(inner, 1)
        return inner

    def original_message_block(self, sender: Person, recipient: Person, locale: str) -> list[Line]:
        r = self.r
        K = KIND["attribution"]
        out: list[Line] = []
        x = r.random()
        if x < 0.4:
            out.append(Line(r.choice(["-----Original Message-----", "-----Original Message-----", "----- Original Message -----", "-----Ursprüngliche Nachricht-----", "-----Message d'origine-----", "-----Mensaje original-----", "-----Messaggio originale-----", "-----Oorspronkelijk bericht-----", "-----原始邮件-----", "-----元のメッセージ-----", "----- Original Message -----", "-----Original Appointment-----"]), K))
        elif x < 0.7:
            out.append(Line(r.choice(["________________________________", "________________________________________", "______________________________________________", "-----------------------------------------------", "________________________________________________________________________"]), K))
        elif x < 0.8:
            out.append(Line(r.choice([" ------------------------------", "------------------------------", "* ------------------------------ *"]), K))
        if r.random() < 0.3:
            out.append(Line("", K))
        out += self.header_block(sender, recipient, K, "outlook", locale)
        return out

    def forward_block(self, sender: Person, recipient: Person, locale: str) -> list[Line]:
        r = self.r
        K = KIND["forward_header"]
        out: list[Line] = []
        marker = r.choice([
            "---------- Forwarded message ---------", "---------- Forwarded message ----------", "---------- Forwarded message ---------",
            "-------- Forwarded Message --------", "-------- Original Message --------", "Begin forwarded message:", "Begin forwarded message:",
            "---------- Weitergeleitete Nachricht ----------", "-------- Weitergeleitete Nachricht --------", "---------- Message transféré ----------",
            "---------- Mensaje reenviado ----------", "---------- Messaggio inoltrato ----------", "---------- Doorgestuurd bericht ----------",
            "---------- Forwarded message ---------", "-----Forwarded Message-----", "----- Forwarded message -----", "-------- 転送メッセージ --------",
            "---------- 转发的邮件 ----------", "FYI - forwarded message below:", "Forwarded message:",
        ])
        out.append(Line(marker, K))
        if marker.startswith("Begin") or r.random() < 0.3:
            out.append(Line("", K))
        out += self.header_block(sender, recipient, K, "forward", locale if r.random() < 0.7 else "en")
        return out

    # --- disclaimers ------------------------------------------------------------------

    def disclaimer_text(self, me: Person) -> list[str]:
        r = self.r
        t = r.choice(V.DISCLAIMERS)
        t = t.format(company=me.company, regno=self.digits(r.choice([6, 7, 8])), address=", ".join(me.address), country=r.choice(["England and Wales", "Scotland", "Ireland", "Delaware", "the Netherlands"]), vat=f"GB{self.digits(9)}", url=me.url, email=r.choice([me.email, f"privacy@{self.domain_for(me.company)}", f"noreply@{self.domain_for(me.company)}"]), year=r.randint(2010, 2026), name1=self.person().full, name2=self.person().full)
        width = r.choice([0, 0, 72, 76, 78, 80, 60, 100])
        if width and len(t) > width:
            return _wrap(t, width)
        return [t]

    def disclaimer(self, me: Person) -> list[Line]:
        r = self.r
        K = KIND["disclaimer"]
        out: list[Line] = []
        x = r.random()
        if x < 0.25:
            footer = r.choice(V.MAILING_LIST_FOOTERS)
            dom = self.domain_for(me.company)
            for t in footer:
                out.append(Line(t.format(list=r.choice(["riak-users", "dev", "users", "announce", "python-dev", "team"]), domain=dom, id=self.alnum(12), org=me.first.lower(), repo=r.choice(["app", "server", "docs", "infra"]), num=r.randint(1, 9999), company=me.company, url=me.url), K))
            return out
        n = r.choices([1, 1, 2, 3], weights=[50, 20, 20, 10])[0]
        for i in range(n):
            if i:
                out.append(Line("", K))
            if r.random() < 0.2:
                out.append(Line(r.choice(["**********************************************************************", "________________________________", "----------------------------------------", "=========================================", "DISCLAIMER", "CONFIDENTIALITY NOTICE", "IMPORTANT NOTICE:", "Legal Notice:", "AVISO LEGAL", "Hinweis:", "Avertissement :"]), K))
            for t in self.disclaimer_text(me):
                out.append(Line(t, K))
        return out

    # --- top-level assembly --------------------------------------------------------------

    def email(self) -> tuple[list[Line], dict[str, object]]:
        r = self.r
        me = self.person()
        other = self.person(me.locale if r.random() < 0.7 else None)
        locale = me.locale
        greetings = V.GREETINGS.get(locale, V.GREETINGS["en"])
        closings = V.CLOSINGS.get(locale, V.CLOSINGS["en"])
        layout = r.choices(
            ["plain", "top", "bottom", "inline", "forward", "outlook", "mobile", "quote_only", "short", "chain"],
            weights=[16, 30, 6, 6, 8, 12, 8, 3, 6, 5],
        )[0]
        lines: list[Line] = []
        meta: dict[str, object] = {"layout": layout, "locale": locale}

        def blank(n: int = 1):
            for _ in range(n):
                lines.append(Line("", KIND["reply"]))

        def greeting_part():
            if r.random() < 0.7:
                lines.append(Line(self.fill(r.choice(greetings), me, other), KIND["greeting"]))
                if r.random() < 0.85:
                    blank()

        def reply_part(max_s: int = 8):
            lines.extend(self.body_lines(me, other, KIND["reply"], max_sentences=max_s))

        def closing_part(force: bool = False) -> bool:
            if force or r.random() < 0.7:
                if r.random() < 0.85:
                    blank()
                c = r.choice(closings)
                lines.append(Line(c, KIND["closing"]))
                return True
            return False

        sig_tagged = False

        def sig_part(prob: float = 0.65):
            nonlocal sig_tagged
            x = r.random()
            if x < prob:
                if r.random() < 0.35:
                    blank()
                lines.extend(self.signature(me, tag=not sig_tagged))
                sig_tagged = True
            elif x < prob + 0.12:
                blank()
                lines.append(Line(r.choice(V.MOBILE_SIGS), KIND["signature"]))

        def disclaimer_part(prob: float = 0.2):
            if r.random() < prob:
                blank(r.choice([1, 1, 2]))
                lines.extend(self.disclaimer(me))

        def quote_part(prefixed: bool | None = None, attrib: bool = True):
            if prefixed is None:
                prefixed = r.random() < 0.7
            if attrib:
                if r.random() < 0.85:
                    blank(r.choice([1, 1, 1, 2, 3]))
                if prefixed or r.random() < 0.3:
                    lines.extend(self.attribution(other, locale))
                    if not prefixed and r.random() < 0.5:
                        blank()
                    elif prefixed and r.random() < 0.25:
                        blank()
                else:
                    lines.extend(self.original_message_block(other, me, locale))
                    if r.random() < 0.7:
                        blank()
            lines.extend(self.quoted_message(other, me, 1, prefixed, locale))

        if layout == "plain":
            greeting_part()
            reply_part()
            closing_part()
            sig_part(0.7)
            disclaimer_part(0.25)
        elif layout == "top":
            greeting_part()
            reply_part()
            closing_part()
            sig_part(0.6)
            quote_part()
            disclaimer_part(0.15)
        elif layout == "bottom":
            quote_part(prefixed=True)
            blank()
            reply_part()
            closing_part()
            sig_part(0.5)
            disclaimer_part(0.1)
        elif layout == "inline":
            lines.extend(self.attribution(other, locale))
            for i in range(r.randint(2, 4)):
                if i:
                    blank()
                q = self.quote_prefix([Line(ln.text, KIND["quote"]) for ln in self.body_lines(other, me, KIND["quote"], max_sentences=3)], 1)
                lines.extend(q)
                blank()
                lines.extend(self.body_lines(me, other, KIND["reply"], max_sentences=3))
            closing_part()
            sig_part(0.5)
        elif layout == "forward":
            if r.random() < 0.7:
                greeting_part()
                reply_part(4)
                closing_part()
                sig_part(0.5)
                blank(r.choice([1, 2]))
            lines.extend(self.forward_block(other, me, locale))
            blank(r.choice([1, 1, 2]))
            fwd = self.quoted_message(other, me, 1, False, locale)
            lines.extend(fwd)
            disclaimer_part(0.1)
        elif layout == "outlook":
            greeting_part()
            reply_part()
            closing_part()
            sig_part(0.7)
            disclaimer_part(0.15)
            blank(r.choice([1, 1, 2]))
            lines.extend(self.original_message_block(other, me, locale))
            if r.random() < 0.8:
                blank()
            lines.extend(self.quoted_message(other, me, 1, r.random() < 0.15, locale))
            disclaimer_part(0.1)
        elif layout == "mobile":
            reply_part(3)
            blank()
            lines.append(Line(r.choice(V.MOBILE_SIGS), KIND["signature"]))
            if r.random() < 0.8:
                quote_part(prefixed=r.random() < 0.85)
        elif layout == "quote_only":
            quote_part(prefixed=r.random() < 0.7)
        elif layout == "short":
            if r.random() < 0.3:
                greeting_part()
            lines.append(Line(self.fill(r.choice(V.BODY_SENTENCES_EN[60:90]), me, other), KIND["reply"]))
            if r.random() < 0.5:
                closing_part()
                sig_part(0.5)
            if r.random() < 0.5:
                quote_part()
        elif layout == "chain":
            greeting_part()
            reply_part(3)
            closing_part()
            sig_part(0.6)
            prefixed = r.random() < 0.5
            for _ in range(r.randint(2, 3)):
                p = self.person(locale if r.random() < 0.7 else None)
                blank()
                if prefixed:
                    lines.extend(self.attribution(p, locale))
                    lines.extend(self.quoted_message(p, me, 1, True, locale))
                else:
                    lines.extend(self.original_message_block(p, me, locale))
                    blank()
                    lines.extend(self.quoted_message(p, me, 1, False, locale))
            disclaimer_part(0.15)
        # noise
        self.noise(lines)
        return lines, meta

    def noise(self, lines: list[Line]) -> None:
        r = self.r
        for i, ln in enumerate(lines):
            if ln.text == "":
                continue
            x = r.random()
            if x < 0.05:
                lines[i] = Line(ln.text + " " * r.randint(1, 3), ln.kind, ln.spans)
            elif x < 0.07 and not ln.spans:
                lines[i] = Line(ln.text.lower(), ln.kind, ln.spans)
            elif x < 0.09:
                # non-breaking space somewhere
                t = ln.text
                j = r.randint(0, len(t))
                lines[i] = Line(t[:j] + " " + t[j:], ln.kind, [(s + (1 if s >= j else 0), e + (1 if e > j else 0), f) for s, e, f in ln.spans])


# ---------------------------------------------------------------------------------
# Rendering + labelling


def render(lines: list[Line], gen: Gen) -> Example:
    r = gen.r
    nl = "\r\n" if r.random() < 0.15 else "\n"
    texts = []
    spans: list[tuple[int, int, str]] = []
    line_kinds: list[int] = []
    offset = 0
    for ln in lines:
        for s, e, f in ln.spans:
            # line spans are code-point offsets; the document uses UTF-16 units
            spans.append((offset + _u16(ln.text[:s]), offset + _u16(ln.text[:e]), f))
        texts.append(ln.text)
        line_kinds.append(-1 if ln.text.strip(" \t") == "" else ln.kind)
        offset += _u16(ln.text) + len(nl)
    text = nl.join(texts)
    if r.random() < 0.5:
        text += nl
    if r.random() < 0.1:
        text += nl * r.randint(1, 3)
        line_kinds.append(-1)
    contact = contact_from_spans(text, spans, lines)
    reply = reply_from_lines(lines)
    return Example(text=text, line_kinds=line_kinds, spans=spans, contact=contact, reply=reply, meta={})


def _u16(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _u16_slice(text: str, start: int, end: int) -> str:
    b = text.encode("utf-16-le")
    return b[start * 2 : end * 2].decode("utf-16-le", errors="ignore")


def contact_from_spans(text: str, spans: list[tuple[int, int, str]], lines: list[Line]) -> dict[str, object]:
    c: dict[str, object] = {}
    for s, e, f in spans:
        val = _u16_slice(text, s, e).strip()
        key = f.lower()
        if f in ("PHONE", "EMAIL", "URL"):
            c.setdefault(key, []).append(val)
        elif f == "ADDRESS":
            c["address"] = (c["address"] + ", " + val) if "address" in c else val
        else:
            c.setdefault(key, val)
    return c


def reply_from_lines(lines: list[Line]) -> str:
    keep = {KIND["reply"], KIND["greeting"], KIND["closing"]}
    out: list[str] = []
    for ln in lines:
        if ln.text.strip(" \t") == "":
            if out and out[-1] != "":
                out.append("")
        elif ln.kind in keep:
            out.append(ln.text.rstrip())
        else:
            if out and out[-1] != "":
                out.append("")
    return "\n".join(out).strip()


def label_tokens(ex: Example) -> tuple[list, np.ndarray, np.ndarray]:
    """Returns (tokens, line_kind_labels[T], bio_labels[T]); -100 = ignored."""
    tokens = tokenize(ex.text)
    T = len(tokens)
    kinds = np.full(T, -100, dtype=np.int64)
    bio = np.zeros(T, dtype=np.int64)
    li = 0
    for i, t in enumerate(tokens):
        k = ex.line_kinds[li] if li < len(ex.line_kinds) else -1
        kinds[i] = k if k >= 0 else -100
        if t.cls == CLASS_NEWLINE:
            li += 1
    # bio
    if ex.spans:
        spans = sorted(ex.spans)
        si = 0
        for i, t in enumerate(tokens):
            while si < len(spans) and spans[si][1] <= t.start:
                si += 1
            if si >= len(spans):
                break
            s, e, f = spans[si]
            if t.end <= s:
                continue
            if t.cls == CLASS_SPACE and t.start == s:
                continue
            # token overlaps span
            first = t.start == s or (i > 0 and tokens[i - 1].end <= s)
            bio[i] = BIO[("B-" if first else "I-") + f]
    return tokens, kinds, bio


# ---------------------------------------------------------------------------------
# Dataset


@dataclass
class Encoded:
    rows: np.ndarray  # [T, NUM_SLOTS] int16
    kinds: np.ndarray  # [T] int8 (-100 → -1)
    bio: np.ndarray  # [T] int8


def encode_example(ex: Example) -> Encoded:
    tokens, kinds, bio = label_tokens(ex)
    rows = np.asarray(featurize_tokens(tokens), dtype=np.int16).reshape(-1, NUM_SLOTS)
    return Encoded(rows, np.where(kinds < 0, -1, kinds).astype(np.int8), bio.astype(np.int8))


def generate(seed: int, n: int, max_tokens: int = 900) -> list[Example]:
    g = EmailGen(seed)
    out: list[Example] = []
    while len(out) < n:
        lines, meta = g.email()
        ex = render(lines, g)
        ex.meta = meta
        if _u16(ex.text) > max_tokens * 3:
            continue
        out.append(ex)
    return out


def build_split(seed: int, n: int, path: Path) -> list[Example]:
    t0 = time.time()
    exs = generate(seed, n)
    rows, kinds, bio, offsets = [], [], [], [0]
    for ex in exs:
        enc = encode_example(ex)
        rows.append(enc.rows)
        kinds.append(enc.kinds)
        bio.append(enc.bio)
        offsets.append(offsets[-1] + len(enc.kinds))
    np.savez_compressed(path, rows=np.concatenate(rows), kinds=np.concatenate(kinds), bio=np.concatenate(bio), offsets=np.asarray(offsets, dtype=np.int64))
    print(f"{path.name}: {n} emails, {offsets[-1]} tokens, {time.time() - t0:.0f}s", file=sys.stderr)
    return exs


EXTERNAL = {
    "email_reply_parser": (
        "https://raw.githubusercontent.com/github/email_reply_parser/master/test/emails/",
        ["correct_sig.txt", "email_1_1.txt", "email_1_2.txt", "email_1_3.txt", "email_1_4.txt", "email_1_5.txt", "email_1_6.txt", "email_1_7.txt", "email_1_8.txt", "email_2_1.txt", "email_2_2.txt", "email_2_3.txt", "email_BlackBerry.txt", "email_bullets.txt", "email_iPhone.txt", "email_long_quote.txt", "email_multi_word_sent_from_my_mobile_device.txt", "email_one_is_not_on.txt", "email_sent_from_my_not_signature.txt", "email_sig_delimiter_in_middle_of_line.txt", "greedy_on.txt", "pathological.txt"],
    ),
    "talon_replies": (
        "https://raw.githubusercontent.com/mailgun/talon/master/tests/fixtures/standard_replies/",
        ["android.eml", "aol.eml", "apple_mail.eml", "apple_mail_2.eml", "comcast.eml", "gmail.eml", "hotmail.eml", "iphone.eml", "outlook.eml", "sparrow.eml", "thunderbird.eml", "yahoo.eml", "iphone_reply_text", "sparrow_reply_text"],
    ),
    "talon_signature": (
        "https://raw.githubusercontent.com/mailgun/talon/master/tests/fixtures/signature/emails/stripped/",
        [f"{n}_{p}" for n in ("camel_case", "jeff", "johndoeexamplecom", "long", "short_url", "sparse") for p in ("body", "signature", "sender")],
    ),
}


def download_external() -> None:
    for name, (base, files) in EXTERNAL.items():
        d = CACHE_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            p = d / f
            if p.exists():
                continue
            try:
                with urllib.request.urlopen(base + f, timeout=30) as resp:  # noqa: S310
                    p.write_bytes(resp.read())
            except Exception as e:  # noqa: BLE001
                print(f"skip {name}/{f}: {e}", file=sys.stderr)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    build_split(1, n_train, CACHE_DIR / "train.npz")
    heldout = build_split(999, 3000, CACHE_DIR / "heldout.npz")
    with (CACHE_DIR / "heldout.json").open("w") as f:
        json.dump([{"text": e.text, "line_kinds": e.line_kinds, "reply": e.reply, "contact": e.contact, "meta": e.meta} for e in heldout], f, ensure_ascii=False)
    download_external()


if __name__ == "__main__":
    main()
