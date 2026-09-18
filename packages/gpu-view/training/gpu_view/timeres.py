"""Relative date resolver. Mirrors src/time.ts exactly (shared fixtures in tests/).

Input is the lower-cased text of a TIME_VALUE span; output is an inclusive
[from, to] pair of ISO dates, or None. Weeks start on Monday. All arithmetic is
on calendar dates (no time zones).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from .lexicon import MONTH_ABBREV, MONTHS, SEASONS

UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month",
         "months": "month", "year": "year", "years": "year", "quarter": "quarter",
         "quarters": "quarter"}
STRIP_LEADING = ("the ", "in ", "on ", "during ", "of ", "within ", "for ", "from ", "since ")


def iso(d: date) -> str:
    return d.isoformat()


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def month_range(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def quarter_range(y: int, q: int) -> tuple[date, date]:
    m = (q - 1) * 3 + 1
    return date(y, m, 1), month_range(y, m + 2)[1]


def week_range(d: date) -> tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)


def year_range(y: int) -> tuple[date, date]:
    return date(y, 1, 1), date(y, 12, 31)


def period(unit: str, today: date, shift: int) -> tuple[date, date]:
    """Calendar period `shift` units away from the one containing today."""
    if unit == "day":
        d = today + timedelta(days=shift)
        return d, d
    if unit == "week":
        return week_range(today + timedelta(days=7 * shift))
    if unit == "month":
        d = add_months(date(today.year, today.month, 1), shift)
        return month_range(d.year, d.month)
    if unit == "quarter":
        q = (today.month - 1) // 3 + shift
        y = today.year + q // 4
        return quarter_range(y, q % 4 + 1)
    return year_range(today.year + shift)


def _month_index(word: str) -> int:
    if word in MONTHS:
        return MONTHS.index(word) + 1
    if word == "sept":
        return 9
    if word in MONTH_ABBREV:
        return MONTH_ABBREV.index(word) + 1
    return 0


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    t = t.strip(" ,.!?;:\"'()")
    changed = True
    while changed:
        changed = False
        for lead in STRIP_LEADING:
            if t.startswith(lead):
                t = t[len(lead):]
                changed = True
    return t.strip()


def resolve(text: str, today: date) -> tuple[str, str] | None:
    t = normalize(text)
    if not t:
        return None
    r = _resolve(t, today)
    return (iso(r[0]), iso(r[1])) if r else None


def _resolve(t: str, today: date) -> tuple[date, date] | None:
    if t == "today" or t == "now":
        return today, today
    if t == "yesterday":
        d = today - timedelta(days=1)
        return d, d
    if t == "tomorrow":
        d = today + timedelta(days=1)
        return d, d
    if t in ("ytd", "year to date"):
        return date(today.year, 1, 1), today
    if t in ("mtd", "month to date"):
        return date(today.year, today.month, 1), today
    if t in ("qtd", "quarter to date"):
        return period("quarter", today, 0)[0], today

    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        try:
            d = date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
        return d, d
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if m and 1 <= int(m[2]) <= 12:
        return month_range(int(m[1]), int(m[2]))
    m = re.fullmatch(r"(19|20)\d{2}", t)
    if m:
        return year_range(int(t))

    m = re.fullmatch(r"(last|past|previous|prior|trailing) (\d+) (\w+)", t)
    if m and m[3] in UNITS:
        n = int(m[2])
        unit = UNITS[m[3]]
        if n < 1:
            return None
        if unit == "day":
            return today - timedelta(days=n - 1), today
        if unit == "week":
            return today - timedelta(days=7 * n - 1), today
        if unit == "month":
            return add_months(today, -n) + timedelta(days=1), today
        if unit == "quarter":
            return add_months(today, -3 * n) + timedelta(days=1), today
        return add_months(today, -12 * n) + timedelta(days=1), today
    m = re.fullmatch(r"(next|coming|upcoming) (\d+) (\w+)", t)
    if m and m[3] in UNITS:
        n = int(m[2])
        unit = UNITS[m[3]]
        if n < 1:
            return None
        start = today + timedelta(days=1)
        if unit == "day":
            return start, today + timedelta(days=n)
        if unit == "week":
            return start, today + timedelta(days=7 * n)
        if unit == "month":
            return start, add_months(today, n)
        if unit == "quarter":
            return start, add_months(today, 3 * n)
        return start, add_months(today, 12 * n)
    m = re.fullmatch(r"(\d+) (\w+) ago", t)
    if m and m[2] in UNITS:
        n = int(m[1])
        unit = UNITS[m[2]]
        if unit == "day":
            d = today - timedelta(days=n)
        elif unit == "week":
            d = today - timedelta(days=7 * n)
        elif unit == "month":
            d = add_months(today, -n)
        elif unit == "quarter":
            d = add_months(today, -3 * n)
        else:
            d = add_months(today, -12 * n)
        return d, d
    m = re.fullmatch(r"(this|current|last|past|previous|prior|next|coming|upcoming) (\w+)", t)
    if m and m[2] in UNITS:
        unit = UNITS[m[2]]
        shift = 0 if m[1] in ("this", "current") else (-1 if m[1] in ("last", "past", "previous", "prior") else 1)
        return period(unit, today, shift)

    m = re.fullmatch(r"q([1-4])(?: (?:of )?(\d{4}))?", t)
    if m:
        return quarter_range(int(m[2]) if m[2] else today.year, int(m[1]))
    m = re.fullmatch(r"h([12])(?: (?:of )?(\d{4}))?", t)
    if m:
        y = int(m[2]) if m[2] else today.year
        return (date(y, 1, 1), date(y, 6, 30)) if m[1] == "1" else (date(y, 7, 1), date(y, 12, 31))
    m = re.fullmatch(r"([a-z]+)(?: (\d{4}))?", t)
    if m:
        mi = _month_index(m[1])
        if mi:
            return month_range(int(m[2]) if m[2] else today.year, mi)
    # "september 10 2026", "september 10, 2026", "sep 10", "10 september 2026", "10th of sep"
    m = re.fullmatch(r"([a-z]+) (\d{1,2})(?:st|nd|rd|th)?,?(?: (\d{4}))?", t)
    if m and _month_index(m[1]):
        return _day(int(m[3]) if m[3] else today.year, _month_index(m[1]), int(m[2]))
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)? (?:of )?([a-z]+),?(?: (\d{4}))?", t)
    if m and _month_index(m[2]):
        return _day(int(m[3]) if m[3] else today.year, _month_index(m[2]), int(m[1]))
    m = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", t)
    if m:
        return _day(int(m[1]), int(m[2]), int(m[3]))
    # seasons (meteorological, northern hemisphere): "this summer", "last winter", "spring 2025"
    m = re.fullmatch(r"(?:(this|last|previous|past|next|coming) )?(spring|summer|autumn|fall|winter)(?: (\d{4}))?", t)
    if m:
        season = m[2]
        if m[3]:
            year = int(m[3])
        else:
            year = today.year
            if season == "winter" and today.month > 2:
                year += 1
            if m[1] in ("last", "previous", "past"):
                year -= 1
            elif m[1] in ("next", "coming"):
                year += 1
        start_m, end_m = SEASONS[season]
        if season == "winter":
            return date(year - 1, 12, 1), month_range(year, 2)[1]
        return date(year, start_m, 1), month_range(year, end_m)[1]
    return None


def _day(y: int, m: int, d: int) -> tuple[date, date] | None:
    try:
        v = date(y, m, d)
    except ValueError:
        return None
    return v, v


if __name__ == "__main__":
    today = date(2026, 9, 17)
    for phrase in ["today", "last 30 days", "this quarter", "last year", "2024", "2025-03-15",
                   "march 2025", "q2 2025", "3 weeks ago", "next 7 days", "last week", "ytd"]:
        print(f"{phrase:<16} {resolve(phrase, today)}")
