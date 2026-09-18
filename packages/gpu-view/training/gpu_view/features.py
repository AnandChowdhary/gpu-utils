"""Sparse feature rows per model token. Must match src/features.ts byte for byte.

The model never sees a field name. Schema membership arrives as anonymous rows:
"this token matched a field of kind K (begin/inside, exact/stem/prefix/typo, via
an alias)", "this token matched an enum value (unique owner? owned by the nearest
preceding/following field?)", and the kind of / distance to the nearest field
match on either side. The only exact word identities are the closed task lexicon;
everything else is a hashed bucket (word + consonant skeleton), shape and length.

Whitespace tokens are dropped before the model: the sequence the model sees is
`match.model_tokens(text)`.
"""

from __future__ import annotations

import unicodedata

from gpu_utils_training.features import CLASS_DIGIT, CLASS_OTHER, SHAPE_TITLE, SHAPE_UPPER, Token, hash_token

from . import match
from .lexicon import KEYWORD_COUNT, KEYWORD_ID

KIND_ID = {"text": 1, "number": 2, "date": 3, "enum": 4, "boolean": 5}
LENGTH_BUCKETS = [1, 2, 3, 4, 6, 8, 12]
WORD_BUCKETS = 128
SKEL_BUCKETS = 64
DIST_ROWS = 6  # 0,1,2,3,4+,none

BLOCKS: list[tuple[str, int]] = [
    ("shape", 8),
    ("length", 8),
    ("word", WORD_BUCKETS),
    ("skeleton", SKEL_BUCKETS),
    ("keyword", KEYWORD_COUNT + 1),
    ("flags", 12),
    ("field_kind", 6),
    ("field_pos", 3),
    ("quality", 6),
    ("alias", 2),
    ("field_neg", 2),
    ("enum_any", 2),
    ("enum_pos", 3),
    ("enum_unique", 2),
    ("enum_prev", 2),
    ("enum_next", 2),
    ("prev_kind", 6),
    ("prev_dist", DIST_ROWS),
    ("next_kind", 6),
    ("next_dist", DIST_ROWS),
    ("position", 4),
]
OFFSET: dict[str, int] = {}
_cursor = 0
for _name, _size in BLOCKS:
    OFFSET[_name] = _cursor
    _cursor += _size
FEATURE_ROWS = _cursor
PADDING_ROW = FEATURE_ROWS
SLOTS = 29

FLAG_HAS_DIGIT, FLAG_ALL_DIGIT, FLAG_PUNCT, FLAG_UPPER, FLAG_FIRST, FLAG_LAST = 0, 1, 2, 3, 4, 5
FLAG_PREV_DIGIT, FLAG_NEXT_DIGIT, FLAG_YEAR, FLAG_SUFFIX, FLAG_ER, FLAG_EST = 6, 7, 8, 9, 10, 11


def length_bucket(n: int) -> int:
    for i, b in enumerate(LENGTH_BUCKETS):
        if n <= b:
            return i
    return len(LENGTH_BUCKETS)


def skeleton(text: str) -> str:
    s = "".join(c for c in text.lower() if c not in "aeiou")
    return s or text.lower()


def dist_bucket(d: int | None) -> int:
    if d is None:
        return DIST_ROWS - 1
    return min(d, DIST_ROWS - 2)


def featurize(text: str, schema: dict) -> tuple[list[Token], list[list[int]]]:
    tokens = match.model_tokens(text)
    n = len(tokens)
    if n == 0:
        return tokens, []
    fields = match.match_spans(tokens, match.field_entries(schema))
    enums = match.match_spans(tokens, match.enum_entries(schema), enum=True)

    field_at: list[match.Span | None] = [None] * n
    for s in fields:
        for i in range(s.start, s.end):
            field_at[i] = s
    enum_at: list[match.Span | None] = [None] * n
    for s in enums:
        for i in range(s.start, s.end):
            enum_at[i] = s

    # Nearest field span strictly before / after each token (excluding its own span).
    prev_span: list[match.Span | None] = [None] * n
    next_span: list[match.Span | None] = [None] * n
    last: match.Span | None = None
    for i in range(n):
        own = field_at[i]
        prev_span[i] = last if (own is None or last is not own) else _before(fields, own)
        if own is not None:
            last = own
    upcoming: match.Span | None = None
    for i in range(n - 1, -1, -1):
        own = field_at[i]
        next_span[i] = upcoming if (own is None or upcoming is not own) else _after(fields, own)
        if own is not None:
            upcoming = own

    rows: list[list[int]] = []
    for i, tok in enumerate(tokens):
        low = tok.text.lower()
        r: list[int] = []

        def add(block: str, v: int) -> None:
            r.append(OFFSET[block] + v)

        add("shape", tok.shape)
        add("length", length_bucket(len(tok.text)))
        add("word", hash_token(tok.text, WORD_BUCKETS))
        add("skeleton", hash_token(skeleton(tok.text), SKEL_BUCKETS))
        add("keyword", KEYWORD_ID.get(low, KEYWORD_COUNT))

        has_digit = any(unicodedata.category(c) == "Nd" for c in tok.text)
        if has_digit:
            add("flags", FLAG_HAS_DIGIT)
        if tok.cls == CLASS_DIGIT:
            add("flags", FLAG_ALL_DIGIT)
        if tok.cls == CLASS_OTHER:
            add("flags", FLAG_PUNCT)
        if tok.shape in (SHAPE_TITLE, SHAPE_UPPER):
            add("flags", FLAG_UPPER)
        if i == 0:
            add("flags", FLAG_FIRST)
        if i == n - 1:
            add("flags", FLAG_LAST)
        if i > 0 and tokens[i - 1].cls == CLASS_DIGIT:
            add("flags", FLAG_PREV_DIGIT)
        if i + 1 < n and tokens[i + 1].cls == CLASS_DIGIT:
            add("flags", FLAG_NEXT_DIGIT)
        if tok.cls == CLASS_DIGIT and len(tok.text) == 4 and tok.text[:2] in ("19", "20"):
            add("flags", FLAG_YEAR)
        if i > 0 and tokens[i - 1].cls == CLASS_DIGIT and low in ("k", "m", "b", "bn", "mm", "%"):
            add("flags", FLAG_SUFFIX)

        if tok.cls == 0 and len(low) >= 5 and (low.endswith("er") or low.endswith("ier")):
            add("flags", FLAG_ER)
        if tok.cls == 0 and len(low) >= 6 and (low.endswith("est") or low.endswith("iest")):
            add("flags", FLAG_EST)

        f = field_at[i]
        add("field_kind", KIND_ID[f.kind] if f else 0)
        add("field_pos", 0 if f is None else (1 if i == f.start else 2))
        add("quality", f.quality + 1 if f else 0)
        add("alias", 1 if (f and f.alias) else 0)
        add("field_neg", 1 if (f and f.neg) else 0)

        e = enum_at[i]
        add("enum_any", 1 if e else 0)
        add("enum_pos", 0 if e is None else (1 if i == e.start else 2))
        add("enum_unique", 1 if (e and len(e.owners) == 1) else 0)
        p, q = prev_span[i], next_span[i]
        add("enum_prev", 1 if (e and p is not None and p.field in e.owners) else 0)
        add("enum_next", 1 if (e and q is not None and q.field in e.owners) else 0)

        add("prev_kind", KIND_ID[p.kind] if p else 0)
        add("prev_dist", dist_bucket(i - p.end if p else None))
        add("next_kind", KIND_ID[q.kind] if q else 0)
        add("next_dist", dist_bucket(q.start - i - 1 if q else None))
        add("position", min(3, (4 * i) // n))
        assert len(r) <= SLOTS
        rows.append(r)
    return tokens, rows


def _before(spans: list[match.Span], own: match.Span) -> match.Span | None:
    best = None
    for s in spans:
        if s.end <= own.start:
            best = s
    return best


def _after(spans: list[match.Span], own: match.Span) -> match.Span | None:
    for s in spans:
        if s.start >= own.end:
            return s
    return None


def pad(rows: list[list[int]]) -> list[list[int]]:
    return [r[:SLOTS] + [PADDING_ROW] * (SLOTS - len(r)) for r in rows]


if __name__ == "__main__":
    print(f"FEATURE_ROWS = {FEATURE_ROWS}, SLOTS = {SLOTS}")
    for name, size in BLOCKS:
        print(f"  {name:<12} {OFFSET[name]:>4} .. {OFFSET[name] + size - 1:>4}")
    schema = {"fields": [
        {"name": "customer", "kind": "text"},
        {"name": "country", "kind": "enum", "values": ["Germany", "France"]},
        {"name": "orders", "kind": "number", "aliases": ["order count"]},
        {"name": "created_at", "kind": "date", "aliases": ["created", "signed up"]},
    ]}
    toks, rows = featurize("customers in Germany or France with more than 5 orders", schema)
    for t, r in zip(toks, rows):
        named = []
        for v in r:
            for name, size in BLOCKS:
                if OFFSET[name] <= v < OFFSET[name] + size:
                    named.append(f"{name}+{v - OFFSET[name]}")
                    break
        print(f"  {t.text:<10} {' '.join(named)}")
