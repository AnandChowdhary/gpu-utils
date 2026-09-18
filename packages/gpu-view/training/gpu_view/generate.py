"""Synthetic query generator: random schemas x hundreds of phrase templates.

Every example carries its text, the schema that produced it, gold roles and
clause boundaries per model token, and the gold view spec. Roles come from the
renderer's own structure, never from a parser. The TypeScript compiler must
round-trip the gold roles back to the gold spec (test/roundtrip.test.ts); if it
cannot, the generator or the compiler is wrong, not the model.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date

from . import match, schema as schema_module
from .schema import BOOL, DATE, ENUM, NUMBER, TEXT, Field, Schema
from .timeres import resolve as resolve_time

ROLES = ["O", "FIELD", "OP", "VALUE", "TIME_VALUE", "CONJ", "NEG", "SORT_FIELD", "SORT_DIR",
         "GROUP_FIELD", "AGG_FN", "AGG_FIELD", "LIMIT", "CHART"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
TODAY = date(2026, 9, 17)

# ------------------------------------------------------------------ word lists
PREFIXES = ["show me", "list", "find", "all", "give me", "show", "get", "which", "what are the",
            "display", "i want to see", "pull up", "search", "find all", "fetch", "look up",
            "filter", "list all", "show all", "can you show", "please list", "i need", "view",
            "show me all", "give me all", "get me", "find me", "let me see", "", "", "", "", ""]
SUFFIXES = ["", "", "", "", "", "please", "thanks", "?", ".", "!", " pls"]
PEOPLE = ["sarah", "arik", "shu", "meera", "tomas", "nadia", "ken", "priya", "john", "alice", "bob",
          "wei", "fatima", "luca", "olga", "diego", "hana", "yusuf", "ingrid", "marco", "amara",
          "chen", "elena", "raj", "sofia", "noah", "leila", "kwame", "anders", "mia"]
SURNAMES = ["smith", "garcia", "mueller", "tanaka", "okafor", "rossi", "kim", "novak", "silva",
            "khan", "berg", "dubois", "patel", "nguyen", "haddad", "larsen", "moreau", "ivanov"]
PHRASES = ["quarterly report", "login bug", "black friday", "onboarding", "roadmap review", "invoice",
           "design spec", "postmortem", "summer sale", "beta launch", "night shift", "north wing",
           "acme corp", "globex", "initech", "vandelay", "umbrella", "hooli", "pied piper",
           "checkout", "dashboard", "mobile app", "newsletter signup", "api timeout", "dark mode",
           "cold brew", "green tea", "alpha", "beta", "gamma", "legacy", "v2", "q3 planning"]
CODES = ["ABC-123", "INV-2049", "SKU-88", "PRJ-7", "X-42", "ORD-1001"]
TRUE_WORDS = ["true", "yes", "on", "enabled"]
FALSE_WORDS = ["false", "no", "off", "disabled"]

EQ_ENUM_OPS = ["is", "=", ":", "equals", "equal to", "set to", "marked as", "in", "from", "of", "with",
               "is", "", "", "", ""]
EQ_TEXT_OPS = ["is", "=", ":", "equals", "named", "called", "titled", "exactly", "is exactly", "set to"]
CONTAINS_OPS = ["contains", "containing", "with", "mentioning", "mentions", "matching", "matches",
                "like", "including", "includes", "has", "", ""]
NEG_WORDS = ["not", "isn't", "is not", "except", "excluding", "other than", "!=", "doesn't", "aren't"]
GT_OPS = ["more than", "greater than", "over", "above", "exceeding", ">", "bigger than", "higher than",
          "larger than"]
# "no less than" is NEG + "less than": the negation flips lt to gte at compile time.
GT_OPS += ["no more than", "not more than", "not over", "not above"]
GTE_OPS = ["at least", "minimum", "min", ">=", "≥", "greater than or equal to"]
LT_OPS = ["less than", "fewer than", "under", "below", "<", "smaller than", "lower than",
          "no less than", "not less than", "not under", "not below"]
LTE_OPS = ["at most", "maximum", "max", "<=", "≤", "up to", "less than or equal to"]
EQ_NUM_OPS = ["=", "is", "equals", "exactly", "equal to", "of", "==", ":"]
BEFORE_OPS = ["before", "prior to", "earlier than", "older than"]
UNTIL_OPS = ["until", "till", "through", "up to", "by", "no later than", "on or before"]
AFTER_OPS = ["after", "later than", "newer than", "past"]
SINCE_OPS = ["since", "from", "starting", "on or after", "starting from"]
IN_DATE_OPS = ["in", "during", "on", "within", "for", "", "", "is", "=", ":"]
EMPTY_OPS = ["without", "missing", "lacking", "with no", "with empty", "with blank", "with missing"]
EMPTY_SUFFIX_OPS = ["is empty", "is blank", "is null", "is missing", "is unset", "empty", "blank",
                    "unknown", "is none", "not set"]
FILLED_OPS = ["with", "with a", "with an", "has", "has a", "having", "have", "have a", "with any"]
FILLED_SUFFIX_OPS = ["is set", "is filled", "is present", "is not empty", "is not blank", "is known",
                     "present", "filled in", "is not null", "is not missing"]
SORT_INTRO = ["sorted by", "sort by", "order by", "ordered by", "arranged by", "ranked by",
              "sorted on", "ranking by", "sort on", "arrange by", "in order of", "ordering by"]
DESC_WORDS = ["descending", "desc", "high to low", "highest first", "descending order",
              "largest first", "biggest first", "z to a", "reverse", "in descending order",
              "from high to low", "highest to lowest", "most first", "decreasing"]
ASC_WORDS = ["ascending", "asc", "low to high", "lowest first", "ascending order",
             "smallest first", "a to z", "in ascending order", "from low to high",
             "lowest to highest", "increasing", "alphabetically", "alphabetical"]
DATE_DESC = ["newest first", "latest first", "most recent first", "newest", "latest", "most recent",
             "recent first"]
DATE_ASC = ["oldest first", "earliest first", "oldest", "earliest"]
GROUP_INTRO = ["by", "per", "grouped by", "group by", "broken down by", "split by", "for each",
               "across", "segmented by", "by", "per", "for every", "bucketed by", "grouping by",
               "aggregated by", "rolled up by"]
UNIT_WORDS = {"day": ["day", "date"], "week": ["week"], "month": ["month"], "quarter": ["quarter"],
              "year": ["year"]}
ADVERB_UNITS = {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter",
                "yearly": "year", "annually": "year"}
SUM_WORDS = ["total", "sum of", "sum", "summed", "total of", "overall", "combined", "aggregate"]
AVG_WORDS = ["average", "avg", "mean", "average of", "mean of", "avg of"]
COUNT_WORDS = ["count of", "number of", "how many", "count", "total number of", "# of", "amount of"]
MIN_WORDS = ["min", "minimum", "lowest", "smallest", "min of", "minimum of"]
MAX_WORDS = ["max", "maximum", "highest", "largest", "max of", "maximum of", "biggest", "peak"]
CHART_WORDS = {
    "bar": ["bar chart", "as a bar chart", "as bars", "in a bar chart", "bar graph", "column chart",
            "as a column chart", "histogram", "as a histogram", "bars", "as a bar graph", "bar"],
    "line": ["line chart", "as a line chart", "line graph", "as a line", "trend line", "as a trend",
             "plotted as a line", "as a line graph", "in a line chart", "as a timeseries",
             "timeseries chart"],
    "pie": ["pie chart", "as a pie", "as a pie chart", "donut chart", "as a donut", "in a pie chart",
            "doughnut chart", "pie"],
    "table": ["as a table", "in a table", "table view", "tabular", "as a grid", "in tabular form",
              "as a list", "table", "in table form", "as rows"],
    "number": ["as a number", "as a single number", "big number", "kpi", "as a kpi", "scorecard",
               "stat tile", "as a metric", "just the number", "as a figure", "as a single value",
               "as a stat", "single number", "as a kpi tile"],
}
LIMIT_TEMPLATES = ["top {n}", "first {n}", "limit {n}", "{n} results", "only {n}", "max {n} rows",
                   "up to {n} results", "show {n}", "{n} rows", "at most {n} results", "just {n}",
                   "limit to {n}", "cap at {n}", "{n} of them", "top {n} only", "first {n} rows",
                   "{n} records", "bottom {n}", "{n} entries", "limit {n} rows"]
FILTER_JOINERS = [" and ", ", ", " with ", " ", " that are ", " which are ", " where ", " having ",
                  " and ", " and ", " and also ", " plus ", " that have ", " whose ", " & "]
FIRST_JOINERS = [" ", " with ", " where ", " that are ", " which have ", " having ", ", ", " that ",
                 " whose ", " which are ", " that were ", " ", " ", " "]
TAIL_JOINERS = [", ", " ", " and ", ", ", " then ", " and then ", " ", ", "]


@dataclass
class Piece:
    text: str
    role: str
    start: int = 0
    end: int = 0


@dataclass
class Clause:
    kind: str  # filter | sort | group | agg | limit | chart | noise
    pieces: list[Piece]
    spec: dict | None = None
    # The parts of the spec that need the rendered text (values) are filled after casing.
    fill: object = None


@dataclass
class Example:
    text: str
    tokens: list[str]
    roles: list[int]
    boundaries: list[int]
    spec: dict
    schema: dict
    now: str
    domain: str


# ------------------------------------------------------------- helpers
def P(text: str, role: str = "O") -> Piece:
    return Piece(text, role)


def entry_words(text: str) -> list[str]:
    return match.words_of(text)


def field_surface(f: Field, rng: random.Random, schema: Schema, fentries: list[match.Entry], fi: int) -> str:
    """A surface form for a field (name, alias, plural, prefix cut or typo) that still resolves."""
    forms = [f.name.replace("_", " ")] + list(f.aliases)
    if rng.random() < 0.15:
        forms.append(f.name)  # raw snake_case
    word = rng.choice(forms)
    if rng.random() < 0.25:
        corrupted = corrupt(word, rng)
        if resolves_to(corrupted, fentries, fi):
            return corrupted
    return word


def resolves_to(text: str, entries: list[match.Entry], fi: int, value: int = -1) -> bool:
    words = entry_words(text)
    if not words:
        return False
    found = match.resolve_words(words, entries)
    if found is None or found.field != fi:
        return False
    if value >= 0 and found.value != value:
        return False
    # The span must consume every word.
    return found.end - found.start >= len(words)


def corrupt(word: str, rng: random.Random) -> str:
    parts = word.split(" ")
    idx = rng.randrange(len(parts))
    w = parts[idx]
    style = rng.random()
    if style < 0.4:
        w2 = w + ("es" if w.endswith(("s", "x", "ch")) else "s")
    elif style < 0.7 and len(w) > match.MIN_PREFIX + 1:
        w2 = w[: len(w) - rng.randint(1, 2)]
    elif len(w) >= match.MIN_TYPO:
        pos = rng.randrange(1, len(w))
        w2 = w[:pos] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[pos + 1 :]
    else:
        return word
    parts[idx] = w2
    return " ".join(parts)


def typo(word: str, rng: random.Random) -> str:
    """Small keyboard-style corruption of a function word; labels are unaffected."""
    if len(word) < 4 or not word.isalpha():
        return word
    pos = rng.randrange(1, len(word) - 1)
    style = rng.random()
    if style < 0.4:
        return word[:pos] + word[pos + 1 :]
    if style < 0.7:
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    return word[:pos] + word[pos] + word[pos:]


def format_number(rng: random.Random, kind_hint: str = "") -> str:
    style = rng.random()
    if style < 0.45:
        return str(rng.choice([1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 90, 100, 150, 200,
                               250, 300, 500, 750, 999, 1000, 2000, 5000]))
    if style < 0.6:
        return str(rng.randint(1, 9999))
    if style < 0.7:
        return f"{rng.randint(1, 99)}.{rng.randint(1, 9)}"
    if style < 0.8:
        return f"{rng.choice([1, 2, 5, 10, 25, 50, 100, 250])}{rng.choice(['k', 'K', 'm', 'M'])}"
    if style < 0.85:
        return f"{rng.randint(1, 9)}.{rng.randint(1, 9)}{rng.choice(['k', 'm'])}"
    if style < 0.92:
        return f"{rng.choice(['$', '€', '£'])}{rng.choice([50, 100, 250, 500, 1000, 1500, 2500, 10000])}"
    if style < 0.96:
        return f"{rng.choice([1, 2, 5, 12, 25, 50, 120])},{rng.choice(['000', '500', '250'])}"
    return f"{rng.randint(1, 99)}%"


NUMBER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(k|m|b|bn|mm)?$")


def parse_number(text: str) -> float | None:
    t = text.lower().replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace(" ", "")
    t = t.rstrip("%")
    m = NUMBER_RE.match(t)
    if not m:
        return None
    v = float(m[1])
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "mm": 1e6}.get(m[2] or "", 1)
    v = v * mult
    return int(v) if v == int(v) else v


def time_phrase(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.12:
        return rng.choice(["today", "yesterday", "tomorrow", "ytd", "now"])
    if style < 0.35:
        n = rng.choice([3, 7, 10, 14, 30, 60, 90, 180, 2, 5, 12, 6])
        unit = rng.choice(["days", "weeks", "months", "years"])
        return f"{rng.choice(['last', 'past', 'previous'])} {n} {unit}"
    if style < 0.6:
        return f"{rng.choice(['this', 'last', 'next', 'current', 'previous'])} {rng.choice(['week', 'month', 'quarter', 'year'])}"
    if style < 0.7:
        return str(rng.randint(2015, 2026))
    if style < 0.8:
        y, m, d = rng.randint(2018, 2026), rng.randint(1, 12), rng.randint(1, 28)
        return f"{y}-{m:02d}-{d:02d}" if rng.random() < 0.8 else f"{y}-{m:02d}"
    if style < 0.9:
        month = rng.choice(["january", "february", "march", "april", "may", "june", "july", "august",
                            "september", "october", "november", "december", "jan", "feb", "mar", "apr",
                            "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        return month if rng.random() < 0.4 else f"{month} {rng.randint(2019, 2026)}"
    if style < 0.95:
        return f"q{rng.randint(1, 4)} {rng.randint(2020, 2026)}" if rng.random() < 0.7 else f"q{rng.randint(1, 4)}"
    n = rng.choice([2, 3, 5, 10, 30, 90])
    return f"{n} {rng.choice(['days', 'weeks', 'months'])} ago"


def text_value(rng: random.Random, f: Field) -> str:
    style = rng.random()
    if f.person and style < 0.5:
        return rng.choice(PEOPLE) if rng.random() < 0.7 else f"{rng.choice(PEOPLE)} {rng.choice(SURNAMES)}"
    if style < 0.3:
        return rng.choice(PEOPLE)
    if style < 0.45:
        return rng.choice(SURNAMES)
    if style < 0.85:
        return rng.choice(PHRASES)
    if style < 0.92:
        return rng.choice(CODES)
    return f"{rng.choice(PEOPLE)}@{rng.choice(['acme', 'globex', 'mail'])}.com"


def maybe_case(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return text
    if r < 0.8:
        return text[0].upper() + text[1:]
    if r < 0.9:
        return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    return text


# ----------------------------------------------------------- clause renderers
def render_filter(schema: Schema, f: Field, rng: random.Random, ctx: dict) -> Clause | None:
    fi = schema.fields.index(f)
    fentries: list[match.Entry] = ctx["fentries"]
    eentries: list[match.Entry] = ctx["eentries"]
    if f.kind == ENUM:
        return render_enum(schema, f, fi, rng, fentries, eentries)
    if f.kind == NUMBER:
        return render_number(schema, f, fi, rng, fentries)
    if f.kind == DATE:
        return render_date(schema, f, fi, rng, fentries)
    if f.kind == BOOL:
        return render_bool(schema, f, fi, rng, fentries, eentries)
    return render_text(schema, f, fi, rng, fentries)


def enum_surface(f: Field, fi: int, value: str, vi: int, rng: random.Random, eentries: list[match.Entry]) -> str:
    text = value
    if rng.random() < 0.08:
        c = corrupt(value, rng)
        if resolves_to(c, eentries, fi, vi):
            text = c
    return maybe_case(text, rng)


def render_enum(schema: Schema, f: Field, fi: int, rng: random.Random, fentries, eentries) -> Clause | None:
    if not f.values:
        return None
    style = rng.random()
    negated = rng.random() < 0.15
    pieces: list[Piece] = []
    fentry_words = {e.words for e in fentries}
    if style < 0.7 or negated and style < 0.85:
        # single value
        vi = rng.randrange(len(f.values))
        value = f.values[vi]
        bare = rng.random() < 0.35 and not negated
        uniquely_owned = sum(1 for e in eentries if e.words == tuple(entry_words(value))) == 1
        collides = tuple(entry_words(value)) in fentry_words
        if bare and uniquely_owned and not collides:
            pieces.append(P(enum_surface(f, fi, value, vi, rng, eentries), "VALUE"))
            return Clause("filter", pieces, {"field": f.name, "op": "eq", "value": value})
        surface = field_surface(f, rng, schema, fentries, fi)
        op = rng.choice(EQ_ENUM_OPS)
        if negated:
            neg = rng.choice(NEG_WORDS)
            layout = rng.random()
            if layout < 0.5:
                pieces += [P(surface, "FIELD"), P(neg, "NEG")]
                if op and op not in ("=", ":", "is", "!="):
                    pieces.append(P(op, "OP"))
            elif layout < 0.75:
                pieces += [P(surface, "FIELD"), P("is", "OP"), P("not", "NEG")]
            else:
                pieces += [P(neg if neg in ("not", "except", "excluding", "other than") else "not", "NEG"),
                           P(surface, "FIELD")]
                if op and op not in ("!=",):
                    pieces.append(P(op, "OP"))
            pieces.append(P(enum_surface(f, fi, value, vi, rng, eentries), "VALUE"))
            return Clause("filter", pieces, {"field": f.name, "op": "neq", "value": value})
        pieces.append(P(surface, "FIELD"))
        if op:
            pieces.append(P(op, "OP"))
        pieces.append(P(enum_surface(f, fi, value, vi, rng, eentries), "VALUE"))
        if rng.random() < 0.1:  # value before field: "urgent priority"
            pieces = [pieces[-1], pieces[0]]
        return Clause("filter", pieces, {"field": f.name, "op": "eq", "value": value})
    # value list
    count = min(len(f.values), rng.choice([2, 2, 2, 3]))
    idxs = rng.sample(range(len(f.values)), count)
    values = [f.values[i] for i in idxs]
    with_field = rng.random() < 0.7
    if not with_field:
        for v in values:
            if sum(1 for e in eentries if e.words == tuple(entry_words(v))) != 1 or tuple(entry_words(v)) in fentry_words:
                with_field = True
    if with_field:
        pieces.append(P(field_surface(f, rng, schema, fentries, fi), "FIELD"))
        if negated:
            pieces.append(P(rng.choice(["not", "is not", "except", "excluding"]), "NEG"))
        intro = rng.choice(["", "", "is", "in", "=", ":", "one of", "any of", "either", "is one of", "is either", "in (", "is any of"])
        if intro:
            pieces.append(P(intro, "OP"))
    else:
        if negated:
            pieces.append(P(rng.choice(["not", "except", "excluding"]), "NEG"))
        if rng.random() < 0.3:
            pieces.append(P(rng.choice(["either", "one of", "any of"]), "OP"))
    sep = rng.choice([" or ", " or ", ", ", " / ", ", ", " and ", " or ", "/"])
    for j, (vi, v) in enumerate(zip(idxs, values)):
        if j > 0:
            if sep.strip():
                pieces.append(P(sep.strip(), "CONJ"))
            if sep == ", " and j == len(values) - 1 and rng.random() < 0.5:
                pieces.append(P("or", "CONJ"))
        pieces.append(P(enum_surface(f, fi, v, vi, rng, eentries), "VALUE"))
    if pieces and pieces[-1].role == "VALUE" and any(p.text == "in (" for p in pieces):
        pieces.append(P(")", "O"))
    if negated:
        # NEG + in: one neq filter per value; keep as one clause with op "nin"-like expansion.
        return Clause("filter", pieces, {"field": f.name, "op": "neq", "value": list(values), "_expand": True})
    return Clause("filter", pieces, {"field": f.name, "op": "in", "value": list(values)})


def render_number(schema: Schema, f: Field, fi: int, rng: random.Random, fentries) -> Clause:
    surface = field_surface(f, rng, schema, fentries, fi)
    style = rng.random()
    pieces: list[Piece] = []
    if style < 0.12:
        a, b = format_number(rng), format_number(rng)
        if (parse_number(a) or 0) > (parse_number(b) or 0):
            a, b = b, a
        layout = rng.random()
        if layout < 0.4:
            pieces = [P(surface, "FIELD"), P("between", "OP"), P(a, "VALUE"), P("and", "OP"), P(b, "VALUE")]
        elif layout < 0.6:
            pieces = [P(surface, "FIELD"), P("from", "OP"), P(a, "VALUE"), P("to", "OP"), P(b, "VALUE")]
        elif layout < 0.8:
            pieces = [P(surface, "FIELD"), P(a, "VALUE"), P(rng.choice(["-", "to", "through"]), "OP"), P(b, "VALUE")]
        else:
            pieces = [P("between", "OP"), P(a, "VALUE"), P("and", "OP"), P(b, "VALUE"), P(surface, "FIELD")]
        fill = {"field": f.name, "op": "between", "_values": 2}
        return Clause("filter", pieces, fill)
    if style < 0.2:
        # presence / absence
        if rng.random() < 0.5:
            op = rng.choice(EMPTY_OPS)
            if op == "with no":
                pieces = [P("with", "OP"), P("no", "NEG"), P(surface, "FIELD")]
                return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
            pieces = [P(op, "OP"), P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
        if rng.random() < 0.3:
            pieces = [P("no", "NEG"), P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
        op = rng.choice(FILLED_OPS)
        words = op.split(" ")
        pieces = [P(words[0], "OP")] + [P(w, "O") for w in words[1:]] + [P(surface, "FIELD")]
        return Clause("filter", pieces, {"field": f.name, "op": "not_empty"})
    negated = rng.random() < 0.1
    fam = rng.choices(["gt", "gte", "lt", "lte", "eq"], weights=[30, 15, 25, 12, 10])[0]
    op = rng.choice({"gt": GT_OPS, "gte": GTE_OPS, "lt": LT_OPS, "lte": LTE_OPS, "eq": EQ_NUM_OPS}[fam])
    value = format_number(rng)
    op_pieces: list[Piece] = []
    for w in op.split(" "):
        op_pieces.append(P(w, "NEG" if w in ("no", "not") else "OP"))
    if negated and not any(p.role == "NEG" for p in op_pieces):  # never stack "not not over"
        op_pieces = [P(rng.choice(["not", "isn't"]), "NEG")] + op_pieces
    layout = rng.random()
    if layout < 0.55:
        pieces = [P(surface, "FIELD")] + op_pieces + [P(value, "VALUE")]
    elif layout < 0.85:
        pieces = op_pieces + [P(value, "VALUE"), P(surface, "FIELD")]
    elif layout < 0.93 and fam in ("gte", "lte") and not negated:
        suffix = rng.choice(["or more", "and up", "or higher"]) if fam == "gte" else rng.choice(["or less", "or fewer", "and under"])
        pieces = [P(value, "VALUE")] + [P(w, "OP") for w in suffix.split(" ")] + [P(surface, "FIELD")]
    elif layout < 0.97 and fam == "gte" and not negated:
        pieces = [P(value, "VALUE"), P("+", "OP"), P(surface, "FIELD")]
    else:
        pieces = [P(surface, "FIELD"), P("of", "O")] + op_pieces + [P(value, "VALUE")]
    negs = sum(1 for p in pieces if p.role == "NEG")
    final = fam
    for _ in range(negs):
        final = NEG_FLIP[final]
    return Clause("filter", pieces, {"field": f.name, "op": final, "_values": 1})


NEG_FLIP = {"eq": "neq", "neq": "eq", "gt": "lte", "lte": "gt", "lt": "gte", "gte": "lt",
            "is_true": "is_false", "is_false": "is_true", "is_empty": "not_empty", "not_empty": "is_empty"}


def render_date(schema: Schema, f: Field, fi: int, rng: random.Random, fentries) -> Clause:
    surface = field_surface(f, rng, schema, fentries, fi)
    style = rng.random()
    pieces: list[Piece] = []
    if style < 0.1:
        # presence / absence
        if rng.random() < 0.5:
            op = rng.choice(["without", "missing", "with no", "no"])
            if op == "with no":
                pieces = [P("with", "OP"), P("no", "NEG"), P(surface, "FIELD")]
            elif op == "no":
                pieces = [P("no", "NEG"), P(surface, "FIELD")]
            else:
                pieces = [P(op, "OP"), P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
        op = rng.choice(["with a", "with", "has a", "having a"])
        words = op.split(" ")
        pieces = [P(words[0], "OP")] + [P(w, "O") for w in words[1:]] + [P(surface, "FIELD")]
        return Clause("filter", pieces, {"field": f.name, "op": "not_empty"})
    if style < 0.2:
        a, b = time_phrase(rng), time_phrase(rng)
        layout = rng.random()
        if layout < 0.5:
            pieces = [P(surface, "FIELD"), P("between", "OP"), P(a, "TIME_VALUE"), P("and", "OP"), P(b, "TIME_VALUE")]
        else:
            pieces = [P(surface, "FIELD"), P("from", "OP"), P(a, "TIME_VALUE"), P("to", "OP"), P(b, "TIME_VALUE")]
        return Clause("filter", pieces, {"field": f.name, "op": "between", "_dates": 2})
    fam = rng.choices(["in", "before", "until", "after", "since"], weights=[45, 15, 8, 15, 17])[0]
    op = rng.choice({"in": IN_DATE_OPS, "before": BEFORE_OPS, "until": UNTIL_OPS, "after": AFTER_OPS, "since": SINCE_OPS}[fam])
    value = time_phrase(rng)
    if value.endswith("ago") and fam == "in":
        op = ""
    bare = (fam == "in" and rng.random() < 0.3 and op in ("", "in", "during", "for")
            and len(schema.of_kind(DATE)) == 1)
    op_pieces = [P(w, "NEG" if w in ("no", "not") else "OP") for w in op.split(" ")] if op else []
    if op == "no later than":
        fam = "after"  # NEG + after → until (lte)
        s_neg = 1
    else:
        s_neg = 0
    if bare:
        pieces = op_pieces + [P(value, "TIME_VALUE")]
        return Clause("filter", pieces, {"field": f.name, "op": fam, "_dates": 1, "_bare": True})
    layout = rng.random()
    if layout < 0.75:
        pieces = [P(surface, "FIELD")] + op_pieces + [P(value, "TIME_VALUE")]
    else:
        pieces = op_pieces + [P(value, "TIME_VALUE"), P(surface, "FIELD")] if fam == "in" else [P(surface, "FIELD")] + op_pieces + [P(value, "TIME_VALUE")]
    return Clause("filter", pieces, {"field": f.name, "op": fam, "_dates": 1, "_neg": s_neg})


def render_bool(schema: Schema, f: Field, fi: int, rng: random.Random, fentries, eentries) -> Clause | None:
    surface = field_surface(f, rng, schema, fentries, fi)
    if any(e.words == tuple(entry_words(surface)) for e in eentries):
        return None  # a bare flag that is also an enum value is unlearnable
    style = rng.random()
    negated = rng.random() < 0.25
    pieces: list[Piece] = []
    if style < 0.5:
        if negated:
            pieces = [P(rng.choice(["not", "non", "isn't"]), "NEG"), P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "is_false"})
        intro = rng.choice(["", "", "", "is", "only", "with", "are", "that are", "which are", "has"])
        if intro:
            words = intro.split(" ")
            pieces = [P(intro, "OP" if intro in ("with", "has") else "O")] if len(words) == 1 else [P(w, "O") for w in words]
        pieces.append(P(surface, "FIELD"))
        return Clause("filter", pieces, {"field": f.name, "op": "is_true"})
    if style < 0.65:
        pieces = [P(rng.choice(["without", "missing"]), "OP"), P(surface, "FIELD")]
        return Clause("filter", pieces, {"field": f.name, "op": "is_false"})
    truth = rng.random() < 0.6
    word = rng.choice(TRUE_WORDS if truth else FALSE_WORDS)
    op = rng.choice(["is", "=", ":", "==", "equals", "set to", ""])
    pieces = [P(surface, "FIELD")]
    if negated:
        pieces.append(P("not", "NEG"))
    if op:
        pieces.append(P(op, "OP"))
    pieces.append(P(word, "VALUE"))
    final = "is_true" if truth else "is_false"
    if negated:
        final = NEG_FLIP[final]
    return Clause("filter", pieces, {"field": f.name, "op": final})


def render_text(schema: Schema, f: Field, fi: int, rng: random.Random, fentries) -> Clause:
    surface = field_surface(f, rng, schema, fentries, fi)
    style = rng.random()
    pieces: list[Piece] = []
    if style < 0.12:
        if rng.random() < 0.5:
            op = rng.choice(EMPTY_OPS)
            if op == "with no":
                pieces = [P("with", "OP"), P("no", "NEG"), P(surface, "FIELD")]
            else:
                pieces = [P(op, "OP"), P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
        if rng.random() < 0.5:
            words = rng.choice(EMPTY_SUFFIX_OPS).split(" ")
            pieces = [P(surface, "FIELD")] + [P(w, "NEG" if w == "not" else "OP") for w in words]
            return Clause("filter", pieces, {"field": f.name, "op": "is_empty"})
        if rng.random() < 0.5:
            op = rng.choice(FILLED_OPS)
            words = op.split(" ")
            pieces = [P(words[0], "OP")] + [P(w, "O") for w in words[1:]] + [P(surface, "FIELD")]
            return Clause("filter", pieces, {"field": f.name, "op": "not_empty"})
        words = rng.choice(FILLED_SUFFIX_OPS).split(" ")
        pieces = [P(surface, "FIELD")] + [P(w, "NEG" if w == "not" else "OP") for w in words]
        return Clause("filter", pieces, {"field": f.name, "op": "not_empty"})
    if f.person and style < 0.35:
        # "assigned to me", "owner is me", "my ..." is not supported; "me" is eq.
        layout = rng.random()
        if layout < 0.5:
            pieces = [P(surface, "FIELD"), P("to", "O"), P("me", "VALUE")]
        elif layout < 0.75:
            pieces = [P(surface, "FIELD"), P(rng.choice(["is", "=", ":"]), "OP"), P("me", "VALUE")]
        else:
            pieces = [P(surface, "FIELD"), P("me", "VALUE")]
        negated = rng.random() < 0.15
        if negated:
            pieces.insert(1, P("not", "NEG"))
            return Clause("filter", pieces, {"field": f.name, "op": "neq", "value": "me"})
        return Clause("filter", pieces, {"field": f.name, "op": "eq", "value": "me"})
    negated = rng.random() < 0.1
    if style < 0.85:
        fam = "eq" if rng.random() < 0.4 else "contains"
        op = rng.choice(EQ_TEXT_OPS if fam == "eq" else CONTAINS_OPS)
        value = maybe_case(text_value(rng, f), rng)
        quoted = rng.random() < 0.15
        val_pieces = [P('"', "O"), P(value, "VALUE"), P('"', "O")] if quoted else [P(value, "VALUE")]
        op_pieces = [P(w, "OP") for w in op.split(" ")] if op else []
        if negated:
            neg = rng.choice(["not", "doesn't", "does not", "isn't", "is not"])
            if fam == "contains" and not op:
                op_pieces = [P("contain", "OP")]
            op_pieces = [P(w, "NEG") for w in neg.split(" ")] + op_pieces
            if fam == "eq" and op in ("is", "is exactly"):
                op_pieces = [P("is", "OP"), P("not", "NEG")]
        layout = rng.random()
        if layout < 0.8 or not op:
            pieces = [P(surface, "FIELD")] + op_pieces + val_pieces
        else:
            pieces = op_pieces + val_pieces + [P(surface, "FIELD")] if fam == "contains" else [P(surface, "FIELD")] + op_pieces + val_pieces
        final = NEG_FLIP[fam] if (negated and fam == "eq") else fam
        if negated and fam == "contains":
            return None  # negated contains is not in the spec's op set; never generate it
        return Clause("filter", pieces, {"field": f.name, "op": final, "_text": True})
    # list of text values
    values = [maybe_case(text_value(rng, f), rng) for _ in range(rng.choice([2, 2, 3]))]
    pieces = [P(surface, "FIELD")]
    if negated:
        pieces.append(P(rng.choice(["not", "is not", "except"]), "NEG"))
    intro = rng.choice(["", "is", "in", "one of", "either", ":", "="])
    if intro:
        pieces.append(P(intro, "OP"))
    sep = rng.choice([" or ", ", ", " or ", " / "])
    for j, v in enumerate(values):
        if j > 0:
            pieces.append(P(sep.strip(), "CONJ"))
        pieces.append(P(v, "VALUE"))
    if negated:
        return Clause("filter", pieces, {"field": f.name, "op": "neq", "_text": True, "_expand": True})
    return Clause("filter", pieces, {"field": f.name, "op": "in", "_text": True})


def render_sort(schema: Schema, rng: random.Random, ctx: dict, with_limit: bool = False) -> Clause | None:
    fentries = ctx["fentries"]
    candidates = [f for f in schema.fields if f.kind in (NUMBER, DATE, TEXT, ENUM)]
    if not candidates:
        return None
    f = rng.choice(candidates)
    fi = schema.fields.index(f)
    surface = field_surface(f, rng, schema, fentries, fi)
    style = rng.random()
    pieces: list[Piece] = []
    if f.kind == DATE and style < 0.3:
        dir_ = "desc" if rng.random() < 0.7 else "asc"
        word = rng.choice(DATE_DESC if dir_ == "desc" else DATE_ASC)
        date_fields = schema.of_kind(DATE)
        if len(date_fields) == 1 and rng.random() < 0.6:
            pieces = [P(word, "SORT_DIR")]
            return Clause("sort", pieces, {"field": f.name, "dir": dir_})
        layout = rng.random()
        if layout < 0.5:
            pieces = [P(word, "SORT_DIR"), P(rng.choice(["by", "on", "using"]), "O"), P(surface, "SORT_FIELD")]
        else:
            pieces = [P(rng.choice(SORT_INTRO), "O"), P(surface, "SORT_FIELD"), P(word, "SORT_DIR")]
        return Clause("sort", pieces, {"field": f.name, "dir": dir_})
    if with_limit:
        n = rng.choice([3, 5, 10, 20, 25, 50, 100])
        top = rng.random() < 0.8
        pieces = [P("top" if top else "bottom", "SORT_DIR"), P(str(n), "LIMIT"),
                  P(rng.choice(["by", "by", "ranked by", "sorted by", "on"]), "O"), P(surface, "SORT_FIELD")]
        if rng.random() < 0.3:
            pieces.insert(2, P(schema.entity, "O"))
        return Clause("sort", pieces, {"field": f.name, "dir": "desc" if top else "asc", "_limit": n})
    dir_word = ""
    dir_ = "asc"
    r = rng.random()
    if r < 0.45:
        dir_word, dir_ = rng.choice(DESC_WORDS), "desc"
    elif r < 0.7:
        dir_word, dir_ = rng.choice(ASC_WORDS), "asc"
    layout = rng.random()
    if layout < 0.6:
        pieces = [P(rng.choice(SORT_INTRO), "O"), P(surface, "SORT_FIELD")]
        if dir_word:
            pieces.append(P(dir_word, "SORT_DIR"))
    elif layout < 0.8 and dir_word:
        pieces = [P(dir_word, "SORT_DIR"), P(rng.choice(["by", "on", "in", "sorted by"]), "O"), P(surface, "SORT_FIELD")]
    elif layout < 0.9:
        pieces = [P(rng.choice(["by", "sorted"]), "O"), P(surface, "SORT_FIELD")]
        if dir_word:
            pieces.append(P(dir_word, "SORT_DIR"))
    else:
        word = rng.choice(["highest", "lowest", "largest", "smallest", "biggest"])
        dir_ = "asc" if word in ("lowest", "smallest") else "desc"
        pieces = [P(word, "SORT_DIR"), P(surface, "SORT_FIELD"), P("first", "SORT_DIR")]
    return Clause("sort", pieces, {"field": f.name, "dir": dir_})


def render_group(schema: Schema, rng: random.Random, ctx: dict, intro_ok: bool = True) -> Clause | None:
    fentries = ctx["fentries"]
    candidates = [f for f in schema.fields if f.kind in (ENUM, BOOL, TEXT)] + schema.of_kind(DATE)
    if not candidates:
        return None
    pieces: list[Piece] = []
    spec: dict = {"fields": [], "granularity": None}
    date_fields = schema.of_kind(DATE)
    if date_fields and rng.random() < 0.3:
        f = rng.choice(date_fields)
        unit = rng.choice(list(UNIT_WORDS))
        if len(date_fields) == 1 and rng.random() < 0.7:
            if rng.random() < 0.3:
                adverb = rng.choice([a for a, u in ADVERB_UNITS.items() if u == unit] or ["monthly"])
                unit = ADVERB_UNITS[adverb]
                pieces = [P(adverb, "GROUP_FIELD")]
            else:
                pieces = [P(rng.choice(["per", "by", "for each", "grouped by", "every"]), "O"), P(rng.choice(UNIT_WORDS[unit]), "GROUP_FIELD")]
            spec["fields"] = [f.name]
            spec["granularity"] = unit
            return Clause("group", pieces, spec)
        fi = schema.fields.index(f)
        surface = field_surface(f, rng, schema, fentries, fi)
        unit_word = rng.choice(UNIT_WORDS[unit])
        # "viewing day" may itself be a field surface; then the unit word is not separable.
        combined = match.resolve_words(entry_words(surface + " " + unit_word), fentries)
        if combined is not None and combined.end - combined.start > len(entry_words(surface)):
            pieces = [P(rng.choice(GROUP_INTRO), "O"), P(surface, "GROUP_FIELD")]
            spec["fields"] = [f.name]
            return Clause("group", pieces, spec)
        pieces = [P(rng.choice(GROUP_INTRO), "O"), P(surface, "GROUP_FIELD"), P(unit_word, "GROUP_FIELD")]
        spec["fields"] = [f.name]
        spec["granularity"] = unit
        return Clause("group", pieces, spec)
    count = 1 if rng.random() < 0.8 else 2
    chosen = rng.sample(candidates, min(count, len(candidates)))
    if intro_ok:
        pieces.append(P(rng.choice(GROUP_INTRO), "O"))
    for j, f in enumerate(chosen):
        fi = schema.fields.index(f)
        if j > 0:
            pieces.append(P(rng.choice(["and", "then", ",", "and by"]), "O"))
        pieces.append(P(field_surface(f, rng, schema, fentries, fi), "GROUP_FIELD"))
        spec["fields"].append(f.name)
    if rng.random() < 0.08:
        pieces = pieces[1:] + [P("breakdown", "O")]
    return Clause("group", pieces, spec)


def render_agg(schema: Schema, rng: random.Random, ctx: dict) -> Clause | None:
    fentries = ctx["fentries"]
    numeric = schema.of_kind(NUMBER)
    fn = rng.choices(["sum", "avg", "count", "min", "max"], weights=[30, 25, 25, 10, 10])[0]
    if fn != "count" and not numeric:
        fn = "count"
    words = {"sum": SUM_WORDS, "avg": AVG_WORDS, "count": COUNT_WORDS, "min": MIN_WORDS, "max": MAX_WORDS}[fn]
    word = rng.choice(words)
    fn_pieces = [P(w, "O" if w == "of" else "AGG_FN") for w in word.split(" ")]
    if fn == "count":
        noun = schema.entity
        if rng.random() < 0.25 and numeric:
            f = rng.choice(numeric)
            fi = schema.fields.index(f)
            pieces = fn_pieces + [P(field_surface(f, rng, schema, fentries, fi), "AGG_FIELD")]
            return Clause("agg", pieces, {"fn": "count", "field": f.name})
        pieces = fn_pieces + ([P(noun, "O")] if rng.random() < 0.85 else [])
        return Clause("agg", pieces, {"fn": "count"})
    f = rng.choice(numeric)
    fi = schema.fields.index(f)
    surface = field_surface(f, rng, schema, fentries, fi)
    layout = rng.random()
    if layout < 0.8:
        pieces = fn_pieces + [P(surface, "AGG_FIELD")]
    else:
        pieces = [P(surface, "AGG_FIELD")] + fn_pieces  # "revenue total"
    return Clause("agg", pieces, {"fn": fn, "field": f.name})


def render_limit(rng: random.Random) -> Clause:
    n = rng.choice([3, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000])
    template = rng.choice(LIMIT_TEMPLATES)
    pieces: list[Piece] = []
    for w in template.split(" "):
        if w == "{n}":
            pieces.append(P(str(n), "LIMIT"))
        elif w in ("top", "bottom"):
            pieces.append(P(w, "SORT_DIR"))
        else:
            pieces.append(P(w, "O"))
    return Clause("limit", pieces, {"limit": n})


def render_chart(rng: random.Random) -> Clause:
    kind = rng.choice(list(CHART_WORDS))
    phrase = rng.choice(CHART_WORDS[kind])
    pieces = [P(w, "O" if w in ("as", "a", "in", "an", "the", "just", "form", "view", "plotted") else "CHART")
              for w in phrase.split(" ")]
    if all(p.role == "O" for p in pieces):
        pieces[-1].role = "CHART"
    return Clause("chart", pieces, {"chart": kind})


# ---------------------------------------------------------------- assembly
def entity_noun(schema: Schema, rng: random.Random) -> str:
    if rng.random() < 0.15:
        texts = [f for f in schema.fields if f.kind == TEXT]
        if texts:
            f = rng.choice(texts)
            return f.name.replace("_", " ") + "s"
    return schema.entity


def build(schema: Schema, rng: random.Random) -> Example | None:
    ctx = {"fentries": match.field_entries(schema.to_json()), "eentries": match.enum_entries(schema.to_json())}
    mode = rng.choices(["list", "metric", "count", "chart"], weights=[50, 25, 12, 13])[0]
    n_filters = rng.choices([0, 1, 2, 3], weights=[12, 45, 30, 13])[0]
    if mode == "list" and n_filters == 0 and rng.random() < 0.5:
        n_filters = 1
    fields = rng.sample(schema.fields, min(n_filters, len(schema.fields)))
    filters: list[Clause] = []
    for f in fields:
        c = render_filter(schema, f, rng, ctx)
        if c is not None:
            filters.append(c)
    tail: list[Clause] = []
    head: list[Clause] = []
    limit_done = False
    if mode in ("metric", "chart"):
        agg = render_agg(schema, rng, ctx)
        if agg:
            head.append(agg)
        if rng.random() < 0.75:
            g = render_group(schema, rng, ctx)
            if g:
                head.append(g)
        if rng.random() < 0.3:
            agg2 = render_agg(schema, rng, ctx)
            if agg2 and agg2.spec != (head[0].spec if head else None):
                head.insert(1 if head else 0, agg2)
    if mode in ("metric", "chart") and head and rng.random() < 0.5 and len(schema.of_kind(DATE)) == 1:
        # "total revenue by region this quarter": a bare time filter right after the head.
        f = schema.of_kind(DATE)[0]
        bare = Clause("filter", [P(time_phrase(rng), "TIME_VALUE")], {"field": f.name, "op": "in", "_dates": 1, "_bare": True})
        if rng.random() < 0.4:
            bare.pieces.insert(0, P(rng.choice(["in", "for", "during"]), "OP"))
        filters.insert(0, bare)
        head.append(None)  # marker: first filter attaches with a plain space
    if mode == "count":
        c = Clause("agg", [P(rng.choice(["how many", "count of", "number of", "count", "total number of"]), "AGG_FN")], {"fn": "count"})
        c.pieces = [P(w, "AGG_FN") for w in c.pieces[0].text.split(" ")]
        head.append(c)
        if rng.random() < 0.4:
            g = render_group(schema, rng, ctx)
            if g:
                tail.append(g)
    if mode == "list":
        r = rng.random()
        if r < 0.15:
            s = render_sort(schema, rng, ctx, with_limit=True)
            if s:
                tail.append(s)
                limit_done = True
        elif r < 0.5:
            s = render_sort(schema, rng, ctx)
            if s:
                tail.append(s)
        if rng.random() < 0.2:
            g = render_group(schema, rng, ctx)
            if g:
                tail.append(g)
    if mode in ("metric", "chart") and rng.random() < 0.35:
        s = render_sort(schema, rng, ctx, with_limit=rng.random() < 0.4)
        if s:
            tail.append(s)
            limit_done = "_limit" in (s.spec or {})
    if not limit_done and rng.random() < (0.25 if mode == "list" else 0.3):
        tail.append(render_limit(rng))
    if rng.random() < (0.15 if mode != "chart" else 0.9):
        tail.append(render_chart(rng))
    rng.shuffle(tail)

    # Order: [prefix] [head] [entity] filters tail   |   [prefix] [head] filters [entity]? tail
    parts: list[tuple[str, Clause | None, str]] = []  # (joiner, clause, literal text)
    prefix = rng.choice(PREFIXES)
    entity = entity_noun(schema, rng)
    show_entity = rng.random() < (0.75 if mode == "list" else 0.5) and None not in head
    if mode == "count":
        show_entity = rng.random() < 0.9
    if prefix:
        parts.append(("", None, prefix))
    glue_first_filter = None in head
    for c in head:
        if c is not None:
            parts.append((" ", c, ""))
    # A bare enum / bare bool adjective before the noun: "open issues".
    adjective: Clause | None = None
    if show_entity and filters and rng.random() < 0.35:
        for c in filters:
            if all(p.role in ("VALUE", "FIELD", "NEG") for p in c.pieces) and len(c.pieces) <= 2:
                adjective = c
                break
    if adjective:
        filters.remove(adjective)
        parts.append((" ", adjective, ""))
        parts.append((" ", None, entity))
    elif show_entity:
        parts.append((" ", None, entity))
    for j, c in enumerate(filters):
        joiner = rng.choice(FIRST_JOINERS if j == 0 else FILTER_JOINERS)
        if j == 0 and glue_first_filter:
            joiner = rng.choice([" ", " ", ", ", " for "])
        if j == 0 and not parts:
            joiner = ""
        parts.append((joiner, c, ""))
    for c in tail:
        parts.append((rng.choice(TAIL_JOINERS), c, ""))
    if not any(c for _, c, _ in parts):
        return None
    suffix = rng.choice(SUFFIXES)

    # Render with offsets.
    text = ""
    clauses: list[Clause] = []
    for k, (joiner, clause, literal) in enumerate(parts):
        if k == 0:
            joiner = ""
        if text and joiner and not joiner.startswith(" ") and joiner != ", ":
            joiner = " " + joiner
        text += joiner
        if clause is None:
            text += literal
            continue
        for m, piece in enumerate(clause.pieces):
            ptext = piece.text
            if piece.role == "O" and rng.random() < 0.06:
                ptext = typo(ptext, rng)
            if m > 0 and not (ptext in (")", ",") or text.endswith("(")):
                text += " "
            piece.start = len(text)
            text += ptext
            piece.end = len(text)
        clauses.append(clause)
    if suffix:
        text += suffix if suffix.startswith((" ", "?", ".", "!")) else " " + suffix
    if " ," in text or "  " in text:
        return None
    r = rng.random()
    if r < 0.12:
        text = text.lower()
    elif r < 0.16:
        text = text.upper()
    elif r < 0.2:
        text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    elif r < 0.55 and text:
        text = text[:1].upper() + text[1:]
    return finish(text, clauses, schema)


def finish(text: str, clauses: list[Clause], schema: Schema) -> Example | None:
    tokens = match.model_tokens(text)
    roles = [0] * len(tokens)
    boundaries = [0] * len(tokens)
    spec: dict = {"filters": [], "sort": [], "groupBy": [], "aggregate": []}
    granularity = None
    for clause in clauses:
        first_role_token = None
        span_start, span_end = None, None
        for piece in clause.pieces:
            if piece.role == "O":
                continue
            for i, t in enumerate(tokens):
                if piece.start <= t.start and t.end <= piece.end:
                    roles[i] = ROLE_ID[piece.role]
                    if first_role_token is None:
                        first_role_token = i
                    span_start = t.start if span_start is None else min(span_start, t.start)
                    span_end = t.end if span_end is None else max(span_end, t.end)
        if first_role_token is None:
            continue
        boundaries[first_role_token] = 1
        span = {"start": span_start, "end": span_end}
        s = clause.spec or {}
        if clause.kind == "filter":
            entry = compile_filter(clause, s, text, span, schema)
            if entry is None:
                return None
            spec["filters"].extend(entry)
        elif clause.kind == "sort":
            spec["sort"].append({"field": s["field"], "dir": s["dir"], "span": span})
            if "_limit" in s:
                spec["limit"] = s["_limit"]
        elif clause.kind == "group":
            runs: list[dict] = []
            gid = ROLE_ID["GROUP_FIELD"]
            for i, t in enumerate(tokens):
                if roles[i] == gid and span_start <= t.start < span_end:
                    if runs and roles[i - 1] == gid:
                        runs[-1]["end"] = t.end
                    else:
                        runs.append({"start": t.start, "end": t.end})
            if len(runs) != len(s["fields"]):
                return None
            for name, run in zip(s["fields"], runs):
                spec["groupBy"].append({"field": name, "span": run})
            if s.get("granularity"):
                granularity = s["granularity"]
        elif clause.kind == "agg":
            item = {"fn": s["fn"], "span": span}
            if "field" in s:
                item["field"] = s["field"]
            spec["aggregate"].append(item)
        elif clause.kind == "limit":
            spec["limit"] = s["limit"]
        elif clause.kind == "chart":
            spec["chart"] = s["chart"]
    if granularity:
        spec["granularity"] = granularity
    return Example(text, [t.text for t in tokens], roles, boundaries, spec, schema.to_json(), TODAY.isoformat(), schema.domain)


def compile_filter(clause: Clause, s: dict, text: str, span: dict, schema: Schema) -> list[dict] | None:
    f = schema.by_name(s["field"])
    values = [text[p.start:p.end] for p in clause.pieces if p.role == "VALUE"]
    times = [text[p.start:p.end] for p in clause.pieces if p.role == "TIME_VALUE"]
    op = s["op"]
    base = {"field": f.name, "op": op, "span": span}
    if f.kind == NUMBER and "_values" in s:
        nums = [parse_number(v) for v in values]
        if any(n is None for n in nums):
            return None
        if op == "between":
            return [{**base, "value": nums}]
        return [{**base, "value": nums[0]}]
    if f.kind == DATE and "_dates" in s:
        ranges = [resolve_time(t, TODAY) for t in times]
        if any(r is None for r in ranges):
            return None
        if op == "between":
            return [{**base, "value": [ranges[0][0], ranges[1][1]]}]
        lo, hi = ranges[0]
        ago = times[0].lower().strip().endswith("ago")
        if s.get("_neg"):
            op = {"after": "until", "since": "before", "before": "since", "until": "after"}[op]
        if op == "in":
            if lo == hi:
                return [{**base, "op": "eq", "value": lo}]
            return [{**base, "op": "between", "value": [lo, hi]}]
        if op == "before":
            return [{**base, "op": "gt" if ago else "lt", "value": hi if ago else lo}]
        if op == "until":
            return [{**base, "op": "gte" if ago else "lte", "value": lo if ago else hi}]
        if op == "after":
            return [{**base, "op": "lt" if ago else "gt", "value": lo if ago else hi}]
        if op == "since":
            return [{**base, "op": "lte" if ago else "gte", "value": hi if ago else lo}]
        return None
    if op in ("is_true", "is_false", "is_empty", "not_empty"):
        return [base]
    if f.kind == ENUM:
        if s.get("_expand"):
            return [{**base, "op": "neq", "value": v} for v in s["value"]]
        return [{**base, "value": s["value"]}]
    # text
    if s.get("value") == "me":
        return [{**base, "value": values[0]}]  # as rendered, e.g. "Me" after casing noise
    if s.get("_expand"):
        return [{**base, "op": "neq", "value": v} for v in values]
    if op == "in":
        return [{**base, "value": values}]
    return [{**base, "value": values[0]}]


def dataset(split: str, count: int, seed: int, domain: str | None = None) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    attempts = 0
    while len(out) < count and attempts < count * 4:
        attempts += 1
        sch = schema_module.sample(split, rng, domain)
        ex = build(sch, rng)
        if ex is None or not ex.tokens or len(ex.tokens) > 48:
            continue
        out.append(ex)
    return out


def example_json(ex: Example) -> dict:
    return {
        "text": ex.text,
        "schema": ex.schema,
        "now": ex.now,
        "domain": ex.domain,
        "tokens": ex.tokens,
        "roles": [ROLES[r] for r in ex.roles],
        "boundaries": ex.boundaries,
        "spec": ex.spec,
    }


if __name__ == "__main__":
    import json
    import sys

    schema_module.assert_disjoint()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for ex in dataset("train", n, seed=7):
        print(ex.text)
        print("  ", " ".join(f"{t}/{ROLES[r]}{'|' if b else ''}" for t, r, b in zip(ex.tokens, ex.roles, ex.boundaries)))
        print("  ", json.dumps(ex.spec))
