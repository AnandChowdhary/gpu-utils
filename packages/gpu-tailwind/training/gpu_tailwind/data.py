"""Synthetic generator: NL phrases -> (token roles, segment boundaries, gold classes).

    uv run python -m gpu_tailwind.data            # writes data/cache/{train,heldout}.jsonl

Every example is built from a structured spec first (families, values, variants),
rendered into text through the lexicon with typos/casing/spelling noise, and the gold
classes come from semantics.emit, which src/compile.ts mirrors.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from gpu_utils_training.features import tokenize

from .lexicon import GLUE_WORDS, NEG_WORDS, PROPS, SEP_WORDS, SPELLING, VALUES, VARIANT_PHRASES, resolve_variant, words_of
from .pairing import compile_pieces
from .semantics import HUES, MOD_OF, SHADES, Value, accepts, emit, parse_value, standalone

LABELS = ["O", "B-PROP", "I-PROP", "B-VAL", "I-VAL", "B-VAR", "I-VAR", "SEP", "NEG"]
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"

# families the generator samples (weight); presets are rarer
FAMILY_WEIGHTS: dict[str, float] = {k: 1.0 for k in PROPS}
for _k in FAMILY_WEIGHTS:
    if _k.startswith("preset:"):
        FAMILY_WEIGHTS[_k] = 0.25
for _k in ("p", "m", "gap", "w", "h", "text", "bg", "border", "rounded", "shadow", "flex", "grid", "justify", "items", "cols", "opacity", "position", "overflow", "z", "transition"):
    FAMILY_WEIGHTS[_k] = 2.5

VARIANT_KEYS = [k for k, v in VARIANT_PHRASES.items() if v]
VARIANT_WEIGHTS = {k: 1.0 for k in VARIANT_KEYS}
for _k in ("hover", "focus", "dark", "md", "lg", "max-sm", "sm", "xl", "active", "disabled", "group-hover"):
    VARIANT_WEIGHTS[_k] = 4.0

REVERSE_SPELLING = {v: k for k, v in SPELLING.items() if k not in ("shadowed", "focussed")}
NUMBER_WORDS = {p: k[4:] for k, ps in VALUES.items() if k.startswith("num:") for p in ps}
VAL_PHRASES: dict[str, list[str]] = {}
for _k, _ps in VALUES.items():
    VAL_PHRASES.setdefault(_k, []).extend(_ps)
MOD_PHRASES = [p for p in MOD_OF]
INT_PHRASES = [p for k, ps in VALUES.items() if k.startswith("int:") for p in ps]
GLUE_BETWEEN = ["of", "is", "set to", ":", "=", "at", "to", "should be", "equal to", "about", "around", "roughly", "-", "—", "like"]
INTROS = ["a card that is", "make it", "i want a", "give me a", "a div with", "button that is", "style it", "the box should be", "make the header", "container with", "a section that's", "i need", "an element with", "nav bar", "modal", "footer", "sidebar", "the hero", "list item", "avatar image", "a panel", "tooltip", "table cell", "a link that is", "the wrapper", "make this", "this should be", "please make it", "let's have", "i'd like", "it should be", "the card is", "header", "footer that is", "a badge", "form", "a heading", "the title", "paragraph", "the image", "icon", "the button", "the input", "toolbar", "menu", "the layout", "page", "the page", "banner", "alert box"]
OUTROS = ["please", "thanks", "if possible", "ok", "for now", "basically", "and that's it", "please and thank you", "!", "."]
LITERALS = ["p-4", "px-6", "py-2", "m-2", "mx-auto", "mt-8", "gap-4", "gap-2", "w-full", "w-1/2", "h-screen", "h-10", "max-w-md", "max-w-prose", "text-sm", "text-lg", "text-2xl", "text-center", "font-bold", "font-medium", "uppercase", "tracking-wide", "leading-relaxed", "truncate", "italic", "underline", "bg-white", "bg-gray-100", "bg-blue-500", "bg-black/50", "text-white", "text-gray-500", "text-red-600", "border", "border-2", "border-gray-200", "rounded", "rounded-lg", "rounded-full", "ring-2", "ring-blue-500", "shadow", "shadow-md", "shadow-lg", "opacity-50", "z-10", "z-50", "relative", "absolute", "fixed", "sticky", "top-0", "inset-0", "overflow-hidden", "overflow-auto", "flex", "inline-flex", "flex-col", "flex-wrap", "items-center", "justify-between", "justify-center", "grid", "grid-cols-3", "col-span-2", "transition", "duration-300", "ease-in-out", "cursor-pointer", "select-none", "hidden", "block", "sr-only", "hover:bg-blue-600", "hover:underline", "focus:ring-2", "focus:outline-hidden", "dark:bg-gray-900", "dark:text-white", "md:flex", "lg:w-1/3", "sm:hidden", "md:grid-cols-2", "lg:text-xl", "group-hover:opacity-100", "disabled:opacity-50", "first:mt-0", "last:border-0", "max-md:hidden", "xl:px-12", "2xl:max-w-7xl", "active:scale-95", "hover:shadow-lg", "focus-visible:ring-2", "motion-reduce:transition-none", "print:hidden"]


LITERALS = [c for c in LITERALS if "-" in c or ":" in c]


@dataclass
class Piece:
    text: str
    role: str  # PROP VAL VAR SEP NEG O
    boundary: bool = False
    key: str | None = None  # prop family for PROP, literal class for literal VAL


@dataclass
class Segment:
    pieces: list[Piece] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    variant: str | None = None


def wchoice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


# ---------------------------------------------------------------- value sampling

def render_number(rng: random.Random, n: str) -> str:
    if n in NUMBER_WORDS.values() and rng.random() < 0.25:
        words = [p for p, v in NUMBER_WORDS.items() if v == n]
        return rng.choice(words)
    return n


def sample_color(rng: random.Random) -> tuple[str, Value]:
    r = rng.random()
    if r < 0.12:
        key = rng.choice([k for k in VAL_PHRASES if k.startswith("col:")])
        return rng.choice(VAL_PHRASES[key]), Value("col", key[4:])
    hue = rng.choice(HUES[:22] + ["blue", "gray", "red", "green", "slate", "white", "black"])
    if hue in ("white", "black"):
        return hue, Value("col", hue)
    r = rng.random()
    if r < 0.45:
        return hue, Value("col", f"{hue}-500")
    if r < 0.65:
        shade = rng.choice(SHADES)
        form = rng.choice([f"{hue} {shade}", f"{hue}-{shade}", f"{hue}{shade}", f"{shade} {hue}"])
        return form, Value("col", f"{hue}-{shade}")
    mod = rng.choice(MOD_PHRASES)
    return f"{mod} {hue}", Value("col", f"{hue}-{MOD_OF[mod]}")


def sample_value_for(rng: random.Random, k: str) -> tuple[str, Value] | None:
    """Pick a phrasing + typed value the family accepts."""
    for _ in range(30):
        r = rng.random()
        if r < 0.22:
            phrase, v = sample_color(rng)
        elif r < 0.45:
            r2 = rng.random()
            if r2 < 0.55:
                n = rng.choice(["0", "0.5", "1", "1.5", "2", "3", "4", "5", "6", "8", "10", "12", "16", "20", "24", "32", "40", "48", "64", "96", "3", "4", "4", "2", "6", "8", "100", "50", "75", "25", "150", "300", "500", "110", "105", "95", "45", "90", "180", "700", "400", "600"])
                phrase, v = render_number(rng, n), Value("num", n)
            elif r2 < 0.75:
                n = rng.choice(["8", "12", "16", "20", "24", "32", "40", "48", "64", "100", "120", "200", "240", "300", "320", "400", "480", "600", "640", "800", "1", "2", "3", "0.5", "1.5", "150", "250"])
                unit = rng.choice(["px", "px", "px", "rem", "em", "vh", "vw", "ms", "s"])
                phrase = rng.choice([f"{n}{unit}", f"{n} {unit}"])
                v = Value("unit", f"{n}{unit}")
            elif r2 < 0.9:
                key = rng.choice([kk for kk in VAL_PHRASES if kk.startswith("frac:")])
                phrase, v = rng.choice(VAL_PHRASES[key]), Value("frac", key[5:])
                pv = parse_value(phrase)
                if pv is None:
                    continue
                v = pv
            else:
                n = rng.choice(["10", "20", "25", "30", "40", "50", "60", "70", "75", "80", "90", "100", "33", "66"])
                phrase, v = f"{n}%", Value("pct", n)
        else:
            key = rng.choice([kk for kk in VAL_PHRASES if not kk.startswith(("int:", "num:", "frac:", "mod:", "col:"))])
            phrase = rng.choice(VAL_PHRASES[key])
            kind, val = key.split(":", 1)
            v = Value(kind, val)
            if kind == "sz" and val in ("xs", "sm", "md", "lg", "xl") and rng.random() < 0.2:
                inten = rng.choice(INT_PHRASES)
                phrase = f"{inten} {phrase}"
                pv = parse_value(phrase)
                if pv is None:
                    continue
                v = pv
        pv = parse_value(phrase)
        if pv is None or pv.kind != v.kind or pv.value != v.value or pv.intensity != v.intensity:
            continue
        if accepts(k, v):
            return phrase, v
    return None


def sample_standalone(rng: random.Random) -> tuple[str, Value] | None:
    for _ in range(30):
        r = rng.random()
        if r < 0.25:
            phrase, v = sample_color(rng)
        else:
            key = rng.choice([kk for kk in VAL_PHRASES if not kk.startswith(("int:", "num:", "frac:", "mod:"))])
            phrase = rng.choice(VAL_PHRASES[key])
            kind, val = key.split(":", 1)
            v = Value(kind, val)
        pv = parse_value(phrase)
        if pv is None or pv.kind != v.kind or pv.value != v.value:
            continue
        if standalone(v, False):
            return phrase, v
    return None


# ---------------------------------------------------------------- units

def unit(rng: random.Random) -> tuple[list[Piece], list[str]]:
    """One (prop, value) unit: pieces and gold classes."""
    r = rng.random()
    if r < 0.04:
        lit = rng.choice(LITERALS)
        return [Piece(lit, "VAL", key=lit)], [lit]
    if r < 0.30:
        s = sample_standalone(rng)
        if s is None:
            return [], []
        phrase, v = s
        neg = v.kind in ("kw", "wt", "rad") and rng.random() < 0.08
        cls = standalone(v, neg)
        if not cls:
            return [], []
        pieces = ([Piece(rng.choice(NEG_WORDS), "NEG")] if neg else []) + [Piece(phrase, "VAL")]
        return pieces, cls
    k = wchoice(rng, FAMILY_WEIGHTS)
    prop_phrase = rng.choice(PROPS[k])
    if k.startswith("preset:"):
        return [Piece(prop_phrase, "PROP", key=k)], emit(k, None, False)
    r = rng.random()
    neg = r < 0.07 and emit(k, None, True) != emit(k, None, False)
    if neg:
        return [Piece(rng.choice(NEG_WORDS), "NEG"), Piece(prop_phrase, "PROP", key=k)], emit(k, None, True)
    if r < 0.25 and emit(k, None, False):
        return [Piece(prop_phrase, "PROP", key=k)], emit(k, None, False)
    s = sample_value_for(rng, k)
    if s is None:
        return ([Piece(prop_phrase, "PROP", key=k)], emit(k, None, False)) if emit(k, None, False) else ([], [])
    phrase, v = s
    cls = emit(k, v, False)
    order = rng.random()
    if v.kind in ("num", "unit", "pct", "frac"):
        prop_first = order < 0.8
    elif v.kind in ("col", "sz", "wt", "mod", "rad"):
        prop_first = order < 0.3
    else:
        prop_first = order < 0.5
    if prop_first:
        pieces = [Piece(prop_phrase, "PROP", key=k)]
        if rng.random() < 0.35:
            pieces.append(Piece(rng.choice(GLUE_BETWEEN), "O"))
        pieces.append(Piece(phrase, "VAL"))
    else:
        pieces = [Piece(phrase, "VAL"), Piece(prop_phrase, "PROP", key=k)]
    return pieces, cls


def apply_variant(classes: list[str], variant: str | None) -> list[str]:
    if variant is None:
        return classes
    if variant == "only-mobile":
        if all(c in ("block", "visible", "flex") for c in classes):
            return ["sm:hidden"]
        return [f"max-sm:{c}" for c in classes]
    if variant == "only-desktop":
        if all(c in ("block", "visible", "flex") for c in classes):
            return ["max-lg:hidden"]
        return [f"lg:{c}" for c in classes]
    return [f"{variant}:{c}" for c in classes]


def segment(rng: random.Random) -> Segment:
    seg = Segment()
    n_units = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
    classes: list[str] = []
    for _ in range(n_units):
        pieces, cls = unit(rng)
        if not pieces:
            continue
        if seg.pieces and rng.random() < 0.15:
            seg.pieces.append(Piece(rng.choice(GLUE_WORDS[:40]), "O"))
        seg.pieces.extend(pieces)
        for c in cls:
            if c not in classes:
                classes.append(c)
    if not seg.pieces:
        return seg
    if rng.random() < 0.32:
        vk = wchoice(rng, VARIANT_WEIGHTS)
        phrase = rng.choice(VARIANT_PHRASES[vk])
        if resolve_variant(words_of(phrase)) == vk:
            seg.variant = vk
            vp = Piece(phrase, "VAR")
            r = rng.random()
            if r < 0.55:
                seg.pieces.append(vp)
            elif r < 0.9:
                seg.pieces.insert(0, vp)
            else:
                seg.pieces.insert(0, vp)
                seg.pieces.insert(1, Piece(rng.choice(["make it", "it becomes", "turn", "switch to", "go", "become", "it should be", "it is", "it gets"]), "O"))
    # gold classes come from the compiler mirror on the clean pieces, so ambiguous
    # pairings resolve identically on both sides
    seg.classes = compile_pieces([(p.role, p.text, p.key) for p in seg.pieces])
    if seg.pieces:
        seg.pieces[0].boundary = True
    return seg


# ---------------------------------------------------------------- noise

VOWELS = "aeiou"


def typo(rng: random.Random, w: str) -> str:
    if len(w) < 5 or not w.isalpha():
        return w
    i = rng.randrange(1, len(w) - 1)
    r = rng.random()
    if r < 0.3:
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    if r < 0.55:
        return w[:i] + w[i + 1 :]
    if r < 0.8:
        return w[:i] + w[i] + w[i:]
    if w[i] in VOWELS:
        return w[:i] + rng.choice(VOWELS.replace(w[i], "")) + w[i + 1 :]
    return w[:i] + w[i + 1] + w[i] + w[i + 2 :]


def noisify(rng: random.Random, text: str, role: str) -> str:
    words = text.split(" ")
    out = []
    typo_done = False
    for w in words:
        if w in REVERSE_SPELLING and rng.random() < 0.2:
            w = REVERSE_SPELLING[w]
        if not typo_done and role in ("PROP", "VAL", "VAR") and rng.random() < 0.05:
            w = typo(rng, w)
            typo_done = True
        out.append(w)
    return " ".join(out)


def casing(rng: random.Random, text: str) -> str:
    r = rng.random()
    if r < 0.08:
        return text.upper()
    if r < 0.2:
        return text[:1].upper() + text[1:]
    if r < 0.25:
        return " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    return text


# ---------------------------------------------------------------- examples

def build(rng: random.Random) -> dict | None:
    n_seg = rng.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.13, 0.07], k=1)[0]
    segs = [s for s in (segment(rng) for _ in range(n_seg)) if s.pieces]
    if not segs:
        return None
    pieces: list[Piece] = []
    if rng.random() < 0.22:
        pieces.append(Piece(rng.choice(INTROS), "O"))
    for i, s in enumerate(segs):
        if i > 0:
            sep = rng.choices(SEP_WORDS[:10], weights=[40, 5, 1, 2, 2, 20, 8, 2, 3, 2], k=1)[0]
            pieces.append(Piece(sep, "SEP"))
        pieces.extend(s.pieces)
    if rng.random() < 0.08:
        pieces.append(Piece(rng.choice(OUTROS), "O"))
    # render text and char spans
    text = ""
    spans: list[tuple[int, int, str, bool]] = []
    for p in pieces:
        t = noisify(rng, p.text, p.role)
        t = casing(rng, t)
        if p.role == "SEP" and t in (",", ";", ".", "!", "/"):
            text = text.rstrip(" ")
            start = len(text)
            text += t + " "
            spans.append((start, start + len(t), p.role, p.boundary))
            continue
        if text and not text.endswith(" "):
            text += " "
        if text and rng.random() < 0.03:
            text += " "  # occasional double space
        start = len(text)
        text += t
        spans.append((start, start + len(t), p.role, p.boundary))
        text += " "
    text = text.strip()
    tokens = tokenize(text)
    labels: list[str] = []
    boundary: list[int] = []
    for tok in tokens:
        lab = "O"
        b = 0
        for start, end, role, sb in spans:
            if tok.start >= start and tok.end <= end:
                if role == "O":
                    lab = "O"
                elif role in ("SEP", "NEG"):
                    lab = role
                else:
                    lab = ("B-" if tok.start == start else "I-") + role
                if sb and tok.start == start:
                    b = 1
                break
        labels.append(lab)
        boundary.append(b)
    classes: list[str] = []
    for s in segs:
        for c in s.classes:
            if c not in classes:
                classes.append(c)
    return {"text": text, "labels": labels, "boundary": boundary, "classes": classes}


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    seen: set[str] = set()
    while len(out) < n:
        ex = build(rng)
        if ex is None or ex["text"] in seen or len(ex["text"]) > 220:
            continue
        seen.add(ex["text"])
        out.append(ex)
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train", 130000, 1), ("heldout", 6000, 2)):
        rows = generate(n, seed)
        with (CACHE / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} examples -> {CACHE / f'{name}.jsonl'}")
    # committed oracle sample for test/oracle.test.ts (generator/compiler parity)
    with (CACHE.parent / "oracle.jsonl").open("w") as f:
        for r in generate(400, 3):
            f.write(json.dumps(r) + "\n")
    for r in generate(12, 7):
        print(r["text"], "=>", " ".join(r["classes"]))


if __name__ == "__main__":
    main()
