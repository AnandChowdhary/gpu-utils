"""Tolerant surface-form matching of query tokens against a schema.

Mirrors src/match.ts exactly. Both the featurizer and the compiler use it: the
featurizer to say "this token matched a field of kind X", the compiler to turn a
FIELD span back into a real field name. If the two disagree the model is scored
against a field the compiler never picks, so there is exactly one implementation
per language and a shared fixture test.

Matching is over n-grams (1..3 words) of *word* tokens (letter or digit runs).
Hyphens and underscores may sit between the words of a span. Quality tiers, best
first: exact, stem (plural/inflection), prefix (>= 4 chars, unique), typo (>= 5
chars, one edit, unique). Multi-word spans only use exact and stem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gpu_utils_training.features import CLASS_DIGIT, CLASS_LETTER, Token, tokenize

EXACT, STEM, INFLECT, PREFIX, TYPO = 0, 1, 2, 3, 4
MIN_PREFIX = 4
MIN_TYPO = 5
MAX_WORDS = 3
CONNECTORS = ("-", "_")

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def words_of(surface: str) -> list[str]:
    """Lower-cased letter/digit runs of a surface form; camelCase and snake_case split."""
    text = _CAMEL.sub(r"\1 \2", surface)
    return [t.text.lower() for t in tokenize(text) if t.cls in (CLASS_LETTER, CLASS_DIGIT)]


def stem(word: str) -> str:
    n = len(word)
    if n >= 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if n >= 5 and word.endswith("sses"):
        return word[:-2]
    if n >= 5 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if n >= 6 and (word.endswith("shes") or word.endswith("ches")):
        return word[:-2]
    if n >= 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def forms(word: str) -> set[str]:
    """Stemmed base forms reachable by stripping comparative/superlative/participle endings.

    "rated" and "rating" share "rat"/"rate"; "cheapest" reaches "cheap"; "downloaded" reaches
    "download". Used by the INFLECT tier so an app can list "cheap" as an alias of price and
    have "cheapest" resolve to it.
    """
    out = {stem(word)}
    n = len(word)
    cands: list[str] = []
    if n >= 6 and word.endswith("iest"):
        cands.append(word[:-4] + "y")
    if n >= 6 and word.endswith("est"):
        cands.append(word[:-3])
    if n >= 5 and word.endswith("ier"):
        cands.append(word[:-3] + "y")
    if n >= 5 and word.endswith("er"):
        cands.append(word[:-2])
    if n >= 5 and word.endswith("ied"):
        cands.append(word[:-3] + "y")
    if n >= 5 and word.endswith("ed"):
        cands.append(word[:-2])
        cands.append(word[:-1])
    if n >= 6 and word.endswith("ing"):
        cands.append(word[:-3])
        cands.append(word[:-3] + "e")
    for c in list(cands):
        if len(c) >= 3 and c[-1] == c[-2] and c[-1] not in "aeiou":
            cands.append(c[:-1])  # bigg → big
    for c in cands:
        if len(c) >= 3:
            out.add(stem(c))
    return out


def within_one_edit(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long[i]:
        i += 1
    return short[i:] == long[i + 1 :]


@dataclass(frozen=True)
class Entry:
    words: tuple[str, ...]
    field: int  # index into schema.fields
    kind: str
    alias: bool  # False for the primary name
    value: int = -1  # enum value index (enum entries only)


@dataclass(frozen=True)
class Span:
    start: int  # model-token index, inclusive
    end: int  # model-token index, exclusive
    field: int
    kind: str
    quality: int
    alias: bool
    value: int = -1
    owners: tuple[int, ...] = ()  # enum entries: every field owning this value


def field_entries(schema: dict) -> list[Entry]:
    out: list[Entry] = []
    for i, f in enumerate(schema["fields"]):
        forms = [(f["name"], False)] + [(a, True) for a in f.get("aliases", [])]
        for surface, alias in forms:
            w = tuple(words_of(surface))
            if w and len(w) <= MAX_WORDS:
                out.append(Entry(w, i, f["kind"], alias))
    return out


def enum_entries(schema: dict) -> list[Entry]:
    out: list[Entry] = []
    for i, f in enumerate(schema["fields"]):
        for j, v in enumerate(f.get("values", [])):
            w = tuple(words_of(v))
            if w and len(w) <= MAX_WORDS:
                out.append(Entry(w, i, f["kind"], False, j))
    return out


def model_tokens(text: str) -> list[Token]:
    """Every token except whitespace/newline runs: the sequence the model sees."""
    return [t for t in tokenize(text) if t.cls not in (2, 3)]


def word_positions(tokens: list[Token]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t.cls in (CLASS_LETTER, CLASS_DIGIT)]


def _connected(tokens: list[Token], a: int, b: int) -> bool:
    """True when every token strictly between indices a and b is a connector."""
    for k in range(a + 1, b):
        if tokens[k].text not in CONNECTORS:
            return False
    return True


def _match_at(
    tokens: list[Token], positions: list[int], wi: int, entries: list[Entry], enum: bool
) -> Span | None:
    lowered = [t.text.lower() for t in tokens]
    for n in range(MAX_WORDS, 0, -1):
        if wi + n > len(positions):
            continue
        idx = positions[wi : wi + n]
        ok = all(_connected(tokens, idx[k], idx[k + 1]) for k in range(n - 1))
        if not ok:
            continue
        span_words = tuple(lowered[i] for i in idx)
        stems = tuple(stem(w) for w in span_words)
        start, end = idx[0], idx[-1] + 1
        for quality in (EXACT, STEM):
            hits = [
                e
                for e in entries
                if len(e.words) == n
                and (e.words == span_words if quality == EXACT else tuple(stem(w) for w in e.words) == stems)
            ]
            if hits:
                first = hits[0]
                owners = tuple(sorted({e.field for e in hits}))
                return Span(start, end, first.field, first.kind, quality, first.alias, first.value, owners)
        if n == 1:
            w = span_words[0]
            if len(w) >= 5:
                qforms = forms(w)
                hits = [e for e in entries if len(e.words) == 1 and qforms & forms(e.words[0])]
                if hits:
                    e = hits[0]
                    owners = tuple(sorted({h.field for h in hits}))
                    return Span(start, end, e.field, e.kind, INFLECT, e.alias, e.value, owners)
            if len(w) >= MIN_PREFIX:
                hits = [e for e in entries if len(e.words) == 1 and len(e.words[0]) >= MIN_PREFIX and e.words[0].startswith(w)]
                keys = {(e.field, e.value) for e in hits}
                if len(keys) == 1:
                    e = hits[0]
                    return Span(start, end, e.field, e.kind, PREFIX, e.alias, e.value, (e.field,))
            if len(w) >= MIN_TYPO:
                hits = [e for e in entries if len(e.words) == 1 and within_one_edit(w, e.words[0])]
                keys = {(e.field, e.value) for e in hits}
                if len(keys) == 1:
                    e = hits[0]
                    return Span(start, end, e.field, e.kind, TYPO, e.alias, e.value, (e.field,))
    return None


def match_spans(tokens: list[Token], entries: list[Entry], enum: bool = False) -> list[Span]:
    """Greedy left-to-right, longest-first matching. Spans never overlap."""
    positions = word_positions(tokens)
    spans: list[Span] = []
    wi = 0
    while wi < len(positions):
        found = _match_at(tokens, positions, wi, entries, enum)
        if found is None:
            wi += 1
            continue
        spans.append(found)
        while wi < len(positions) and positions[wi] < found.end:
            wi += 1
    return spans


def resolve_words(words: list[str], entries: list[Entry]) -> Span | None:
    """Resolve an already-extracted phrase (compiler side). Returns a span with fake indices."""
    text = " ".join(words)
    toks = model_tokens(text)
    positions = word_positions(toks)
    if not positions:
        return None
    return _match_at(toks, positions, 0, entries, False)
