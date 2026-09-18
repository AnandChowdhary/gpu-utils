"""Merged lexicon shared by the generator (Python) and, via src/table.json, the compiler (TS).

Variants are resolved by keyword rules (VAR_RULES) rather than by phrase lookup so
that the shipped table stays small and unseen phrasings still resolve.
"""

from __future__ import annotations

import re

from .lex_props import PROPS
from .lex_values import VALUES
from .lex_variants import GLUE, NEGATIONS, PRESETS, SEPARATORS, VARIANTS

MAX_VARIANT_PHRASES = 40
VARIANT_PHRASES: dict[str, list[str]] = {k: v[:MAX_VARIANT_PHRASES] for k, v in VARIANTS.items()}
GLUE_WORDS: list[str] = GLUE[:110]
NEG_WORDS: list[str] = ["no", "not", "without", "never", "remove", "removes", "removed", "drop", "kill", "zero", "disable", "get rid of", "lose", "strip", "avoid", "prevent", "suppress", "omit", "turn off"]
SEP_WORDS: list[str] = SEPARATORS

# British -> American spellings applied before lookup (both sides).
SPELLING: dict[str, str] = {
    "colour": "color",
    "colours": "colors",
    "grey": "gray",
    "greys": "grays",
    "greyish": "grayish",
    "centre": "center",
    "centred": "centered",
    "centres": "centers",
    "capitalise": "capitalize",
    "capitalised": "capitalized",
    "italicised": "italicized",
    "focussed": "focused",
    "greyscale": "grayscale",
    "behaviour": "behavior",
    "shadowed": "shadow",
}

# Variant keyword rules. Evaluated by resolve_variant on the normalised word list of
# a VARIANT span. Mirrored exactly by resolveVariant in src/compile.ts.
VAR_RULES: dict[str, object] = {
    "tiers": {
        "sm": ["mobile", "phone", "phones", "handheld", "handhelds", "smartphone", "smartphones", "cellphone", "cellphones", "iphone", "android", "small", "tiny", "narrow", "smallest", "xs", "little", "cell"],
        "md": ["tablet", "tablets", "medium", "ipad", "md", "768", "768px", "mid"],
        "lg": ["desktop", "desktops", "laptop", "laptops", "large", "big", "lg", "1024", "1024px", "pc", "computer", "computers", "wide", "displays", "full"],
        "xl": ["xl", "huge", "1280", "1280px", "monitor", "monitors", "widescreen", "giant", "xlarge"],
        "2xl": ["2xl", "1536", "1536px", "ultra", "ultrawide", "massive", "4k", "xxl", "biggest", "largest", "enormous", "widest", "super", "double"],
    },
    # bare breakpoint names default to the mobile-first (min-width) form
    "smLiteral": ["sm", "640", "640px"],
    # words that make a breakpoint its max-* form; "up to" is handled in code
    "max": ["below", "under", "smaller", "narrower", "less", "max", "until", "before", "down", "downwards", "downward", "except", "but", "non", "touch"],
    # words that force the min-* form
    "min": ["up", "above", "over", "least", "+", "upwards", "upward", "beyond", "past", "wider", "from", "larger", "bigger"],
    "xlWords": ["very", "extra"],
    "lgFallback": ["larger", "bigger", "wider"],
    "only": ["only", "exclusively", "just", "solely", "exclusive", "restricted", "limited"],
    "focusWords": ["focus", "focused", "focussed", "focusing", "tabbed"],
    # [variant, keywords, required-extra-keywords]; strong states win over breakpoints
    "strong": [
        ["group-focus", ["group", "parent", "container", "ancestor", "wrapper"], ["focus", "focused", "focussed", "focusing"]],
        ["group-hover", ["group", "parent", "container", "ancestor", "wrapper"], ["hover", "hovered", "hovering", "moused", "mouseover"]],
        ["dark", ["dark", "night", "darkmode"], []],
        ["focus-visible", ["keyboard", "visible", "tabbed"], ["focus", "focused", "focussed", "tabbed"]],
        ["focus-within", ["within", "child", "inside", "descendant", "inner", "input"], ["focus", "focused", "focussed"]],
        ["hover", ["hover", "hovered", "hovering", "mouseover", "rollover", "moused", "mouse", "cursor"], []],
        ["focus", ["focus", "focused", "focussed"], []],
        ["active", ["active", "pressed", "press", "click", "clicked", "clicking", "tap", "tapped", "mousedown"], []],
        ["disabled", ["disabled", "inactive", "unavailable", "greyed", "grayed", "off", "enabled"], []],
        ["print", ["print", "printing", "printed", "printer", "paper", "printout"], []],
        ["checked", ["checked", "ticked", "selected", "toggled", "marked", "chosen", "picked", "switched"], []],
        ["placeholder", ["placeholder", "hint", "ghost"], []],
        ["visited", ["visited", "visiting", "seen"], []],
        ["motion-reduce", ["motion", "animations"], []],
        ["rtl", ["rtl", "arabic", "hebrew", "mirrored"], []],
        ["ltr", ["ltr"], []],
        ["landscape", ["landscape", "sideways"], []],
        ["portrait", ["portrait", "upright"], []],
        ["invalid", ["invalid", "error", "validation", "fails", "failed"], []],
        ["required", ["required", "mandatory", "compulsory"], []],
    ],
    "weak": [
        ["odd", ["odd"], []],
        ["even", ["even", "alternate", "alternating", "zebra", "striped", "stripes", "banded", "other", "second"], []],
        ["first", ["first", "leading", "initial", "beginning", "starting"], []],
        ["last", ["last", "trailing", "final", "ending"], []],
        ["before", ["before", "prepended", "prefix"], []],
        ["after", ["after", "appended", "suffix"], []],
        ["empty", ["empty", "blank", "childless"], []],
        ["open", ["open", "opened", "expanded", "unfolded"], []],
    ],
}


def normalize_word(w: str) -> str:
    w = w.lower()
    return SPELLING.get(w, w)


def words_of(text: str) -> list[str]:
    """Split a span into lowercase words; every punctuation character is its own word."""
    return [normalize_word(w) for w in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())]


def _match(ws: set[str], rules: list) -> str | None:  # type: ignore[type-arg]
    for variant, keys, extra in rules:
        if any(k in ws for k in keys) and (not extra or any(k in ws for k in extra)):
            return variant
    return None


def resolve_variant(words: list[str]) -> str | None:
    """Mirror of resolveVariant in src/compile.ts."""
    r = VAR_RULES
    ws = set(words)
    if "right" in ws and "left" in ws:
        return "rtl" if words.index("right") < words.index("left") else "ltr"
    strong = _match(ws, r["strong"])  # type: ignore[arg-type]
    if strong:
        return strong
    tiers: dict[str, list[str]] = r["tiers"]  # type: ignore[assignment]
    tier: str | None = None
    literal = False
    for name in ("2xl", "xl", "lg", "md", "sm"):
        if any(k in ws for k in tiers[name]):
            tier = name
            break
    if tier is None and any(k in ws for k in r["smLiteral"]):  # type: ignore[operator]
        tier, literal = "sm", True
    if tier is None and any(k in ws for k in r["lgFallback"]):  # type: ignore[operator]
        tier = "lg"
    if tier is None:
        return _match(ws, r["weak"])  # type: ignore[arg-type]
    if tier == "lg" and any(w in ws for w in r["xlWords"]):  # type: ignore[operator]
        tier = "xl"
    if "very" in ws and "small" in ws:
        tier = "sm"
    up_to = any(words[i] == "up" and words[i + 1] == "to" for i in range(len(words) - 1))
    is_max = up_to or any(w in ws for w in r["max"])  # type: ignore[operator]
    is_min = (not up_to and "up" in ws) or any(w in ws and w != "up" for w in r["min"])  # type: ignore[operator]
    if any(w in ws for w in r["only"]):  # type: ignore[operator]
        if tier == "sm":
            return "only-mobile"
        if tier in ("lg", "xl", "2xl"):
            return "only-desktop"
    if tier == "sm" and not literal:
        return "sm" if (is_min and not is_max) else "max-sm"
    return f"max-{tier}" if (is_max and not is_min) else tier


def all_phrases() -> dict[str, tuple[str, str]]:
    """phrase -> (role, key) for every lexicon phrase; used by the generator."""
    out: dict[str, tuple[str, str]] = {}
    for key, phrases in PROPS.items():
        for p in phrases:
            out[p] = ("PROP", key)
    for key, phrases in VALUES.items():
        for p in phrases:
            out.setdefault(p, ("VAL", key))
    return out


__all__ = [
    "PROPS",
    "VALUES",
    "VARIANT_PHRASES",
    "PRESETS",
    "SEP_WORDS",
    "NEG_WORDS",
    "GLUE_WORDS",
    "SPELLING",
    "VAR_RULES",
    "normalize_word",
    "words_of",
    "resolve_variant",
    "all_phrases",
]
