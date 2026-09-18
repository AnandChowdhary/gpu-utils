"""Segment compiler mirror: pairs values with properties exactly like src/decode.ts.

The generator derives gold classes by running this on the clean (pre-noise) pieces of a
segment, so gold and compiler agree even on ambiguous phrasings.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexicon import resolve_variant, words_of
from .semantics import Value, emit, parse_value, standalone

VARIANT_ORDER = ["sm", "md", "lg", "xl", "2xl", "max-sm", "max-md", "max-lg", "max-xl", "print", "motion-reduce", "landscape", "portrait", "rtl", "ltr", "dark", "group-hover", "group-focus", "first", "last", "odd", "even", "empty", "open", "checked", "required", "invalid", "disabled", "visited", "hover", "focus", "focus-visible", "focus-within", "active", "placeholder", "before", "after"]
NUMERIC = ("num", "unit", "pct", "frac")


@dataclass
class Unit:
    index: int
    key: str | None
    value: Value | None
    neg: bool


def apply_variants(classes: list[str], variants: list[str]) -> list[str]:
    """Mirror of applyVariants in compile.ts."""
    if not variants:
        return classes
    only = next((v for v in variants if v.startswith("only-")), None)
    if only:
        others = [v for v in variants if v != only]
        trivial = all(c in ("block", "visible", "flex") for c in classes)
        if trivial:
            return apply_variants(["sm:hidden" if only == "only-mobile" else "max-lg:hidden"], others)
        return apply_variants(classes, [*others, "max-sm" if only == "only-mobile" else "lg"])
    ordered = sorted(set(variants), key=VARIANT_ORDER.index)
    prefix = ":".join(ordered)
    return [f"{prefix}:{c}" for c in classes]


def compile_pieces(pieces: list[tuple[str, str, str | None]]) -> list[str]:
    """pieces: (role, clean_text, key) with role in PROP VAL VAR NEG O; key is the prop
    family for PROP pieces, the literal class for literal VAL pieces, else None."""
    variants: list[str] = []
    props: list[Unit] = []
    vals: list[Unit] = []
    pending = False
    idx = 0
    for role, text, key in pieces:
        if role == "O":
            continue
        if role == "NEG":
            pending = True
            continue
        if role == "VAR":
            v = resolve_variant(words_of(text))
            if v:
                variants.append(v)
            idx += 1
            continue
        if role == "PROP":
            props.append(Unit(idx, key, None, pending))
        else:
            value = parse_value(text) or (Value("lit", key) if key else None)
            if value is not None:
                vals.append(Unit(idx, None, value, pending))
        pending = False
        idx += 1
    assigned: dict[int, list[Unit]] = {}
    loose: list[Unit] = []
    for v in vals:
        best: Unit | None = None
        best_dist = 10**9
        if v.value is not None and v.value.kind != "lit":
            for p in props:
                if not emit(p.key or "", v.value, False):
                    continue
                before = p.index < v.index
                dist = abs(p.index - v.index)
                prefer = before if v.value.kind in NUMERIC else not before
                if dist < best_dist or (dist == best_dist and prefer):
                    best, best_dist = p, dist
        if best is not None:
            assigned.setdefault(best.index, []).append(v)
        else:
            loose.append(v)
    classes: list[str] = []

    def push(cs: list[str]) -> None:
        for c in cs:
            if c not in classes:
                classes.append(c)

    for u in sorted(props + loose, key=lambda u: u.index):
        if u.key is not None:
            values = assigned.get(u.index, [])
            if not values:
                push(emit(u.key, None, u.neg))
            else:
                for v in values:
                    push(emit(u.key, v.value, u.neg or v.neg))
        else:
            push(standalone(u.value, u.neg))  # type: ignore[arg-type]
    if not classes:
        return []
    return apply_variants(classes, variants)
