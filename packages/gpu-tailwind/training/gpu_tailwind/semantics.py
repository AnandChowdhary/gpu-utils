"""Reference semantics: VAL span text -> typed value; (prop, value) -> classes.

src/compile.ts mirrors this module line for line; test/oracle.test.ts checks the
TypeScript compiler reproduces the generator's gold classes from gold tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import PRESETS, PROPS, VALUES, words_of

HUES = ["red", "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose", "slate", "gray", "zinc", "neutral", "stone", "mauve", "olive", "mist", "taupe"]
SHADES = ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"]
SIZES = ["xs", "sm", "md", "lg", "xl", "2xl", "3xl", "4xl", "5xl", "6xl", "7xl", "8xl", "9xl"]
WEIGHTS = ["thin", "extralight", "light", "normal", "medium", "semibold", "bold", "extrabold", "black"]

PROP_OF: dict[str, str] = {p: k for k, ps in PROPS.items() for p in ps}
VAL_OF: dict[str, str] = {}
MOD_OF: dict[str, str] = {}
for _k, _ps in VALUES.items():
    for _p in _ps:
        if _k.startswith("mod:"):
            MOD_OF.setdefault(_p, _k[4:])
        else:
            VAL_OF.setdefault(_p, _k)

SPACING = ["p", "px", "py", "pt", "pr", "pb", "pl", "m", "mx", "my", "mt", "mr", "mb", "ml", "gap", "gap-x", "gap-y", "space-x", "space-y"]
INSETS = ["top", "bottom", "left", "right", "inset"]
SIZING = ["w", "h", "size", "min-w", "max-w", "min-h", "max-h"]
COLORS = ["text", "bg", "border", "border-t", "border-b", "border-l", "border-r", "border-x", "border-y", "ring", "outline", "shadow"]
SZ_SPACING = {"none": "0", "xs": "1", "sm": "2", "md": "4", "lg": "8", "xl": "12", "2xl": "16", "3xl": "24", "4xl": "32", "5xl": "40", "6xl": "48", "7xl": "64"}
SZ_HEIGHT = {"none": "0", "xs": "8", "sm": "12", "md": "16", "lg": "24", "xl": "32", "2xl": "48", "3xl": "64", "4xl": "96"}
SZ_DURATION = {"xs": "75", "sm": "150", "md": "300", "lg": "500", "xl": "700", "2xl": "1000"}
SZ_OPACITY = {"none": "0", "xs": "10", "sm": "25", "md": "50", "lg": "75", "xl": "90", "full": "100"}
WEIGHT_NUM = {"100": "thin", "200": "extralight", "300": "light", "400": "normal", "500": "medium", "600": "semibold", "700": "bold", "800": "extrabold", "900": "black"}


@dataclass
class Value:
    kind: str  # sz kw col mod wt rad num frac pct unit spec lit
    value: str
    intensity: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}:{self.value}{'+' if self.intensity > 0 else '-' if self.intensity < 0 else ''}"


def shift_size(sz: str, by: int, lo: str = "xs", hi: str = "3xl") -> str:
    if sz not in SIZES or by == 0:
        return sz
    i = max(SIZES.index(lo), min(SIZES.index(hi), SIZES.index(sz) + by))
    return SIZES[i]


def shift_shade(shade: str, by: int) -> str:
    i = max(0, min(len(SHADES) - 1, SHADES.index(shade) + 2 * by))
    return SHADES[i]


def parse_color(words: list[str]) -> Value | None:
    """[modifiers] hue [shade] | hue-shade | shade hue."""
    if not words:
        return None
    joined = "".join(words)
    m = re.fullmatch(r"([a-z]+)-?(\d{2,3})", joined)
    if m and m.group(1) in HUES and m.group(2) in SHADES:
        return Value("col", f"{m.group(1)}-{m.group(2)}")
    ws = list(words)
    shade = None
    if ws and ws[-1] in SHADES:
        shade = ws.pop()
    elif ws and ws[0] in SHADES:
        shade = ws.pop(0)
    if not ws:
        return None
    hue = ws[-1]
    if hue not in HUES:
        return None
    mods = ws[:-1]
    if mods:
        mod = MOD_OF.get(" ".join(mods))
        if mod is None:
            return None
        shade = shade or mod
    return Value("col", f"{hue}-{shade or '500'}")


def parse_value(text: str) -> Value | None:
    """Mirror of parseValue in src/compile.ts."""
    words = words_of(text)
    if not words:
        return None
    joined = "".join(words)
    phrase = " ".join(words)
    if phrase in VAL_OF:
        k, v = VAL_OF[phrase].split(":", 1)
        return Value(k, v)
    if phrase in MOD_OF:
        return Value("mod", MOD_OF[phrase])
    # intensifiers
    intensity = 0
    rest = words
    while rest:
        two = " ".join(rest[:2])
        one = rest[0]
        if two in VAL_OF and VAL_OF[two].startswith("int:"):
            intensity += int(VAL_OF[two][4:])
            rest = rest[2:]
        elif one in VAL_OF and VAL_OF[one].startswith("int:"):
            intensity += int(VAL_OF[one][4:])
            rest = rest[1:]
        else:
            break
    if intensity and rest:
        inner = parse_value(" ".join(rest))
        if inner is not None:
            inner.intensity += intensity
            return inner
        return None
    if joined in VAL_OF:
        k, v = VAL_OF[joined].split(":", 1)
        return Value(k, v)
    col = parse_color(words)
    if col is not None:
        return col
    if re.fullmatch(r"\d+(\.\d+)?", joined):
        return Value("num", joined)
    if re.fullmatch(r"\d+(\.\d+)?(px|rem|em|vh|vw|ch|ms|s)", joined):
        return Value("unit", joined)
    if re.fullmatch(r"\d+%", joined):
        return Value("pct", joined[:-1])
    if re.fullmatch(r"\d+/\d+", joined):
        return Value("frac", joined)
    if len(words) == 2 and words[0] in ("light", "dark", "pale", "deep") and words[1] in ("gray", "grey"):
        return Value("col", f"gray-{MOD_OF[words[0]]}")
    return None


def is_int(x: str, lo: int, hi: int) -> bool:
    return bool(re.fullmatch(r"\d+", x)) and lo <= int(x) <= hi


def is_spacing_num(x: str) -> bool:
    if re.fullmatch(r"\d+", x):
        return int(x) <= 96
    return bool(re.fullmatch(r"\d+\.5", x)) and float(x) <= 12


LENGTH_UNITS = ("px", "rem", "em", "vh", "vw", "ch")
TIME_UNITS = ("ms", "s")


def unit_ok(v: Value, units: tuple[str, ...]) -> bool:
    return v.kind == "unit" and v.value.endswith(units)


def spacing_value(k: str, v: Value | None, neg: bool) -> list[str]:
    if neg:
        return [f"{k}-0"]
    if v is None:
        return [f"{k}-0"] if k in INSETS else [f"{k}-4"]
    if v.kind == "num":
        return [f"{k}-{v.value}"] if is_spacing_num(v.value) else []
    if v.kind == "unit":
        return [f"{k}-[{v.value}]"] if unit_ok(v, LENGTH_UNITS) else []
    if v.kind == "pct":
        return [f"{k}-[{v.value}%]"]
    if v.kind == "frac":
        return [f"{k}-{v.value}"] if k in INSETS else []
    if v.kind == "sz":
        if v.value == "full":
            return [f"{k}-full"] if k in INSETS else []
        s = shift_size(v.value, v.intensity, "xs", "7xl") if v.value != "none" else "none"
        n = SZ_SPACING.get(s)
        return [f"{k}-{n}"] if n else []
    if v.kind == "kw":
        if v.value == "auto" and (k.startswith("m") or k in INSETS):
            return [f"{k}-auto"]
        if v.value == "negative":
            return [f"-{k}-4"]
        if v.value in ("top", "bottom", "left", "right") and k in INSETS:
            return [f"{k}-0"]
    return []


def sizing_value(k: str, v: Value | None, neg: bool) -> list[str]:
    if neg:
        return [f"{k}-0"]
    if v is None:
        return {"w": ["w-full"], "h": ["h-full"], "size": ["size-full"], "min-w": ["min-w-0"], "max-w": ["max-w-full"], "min-h": ["min-h-full"], "max-h": ["max-h-full"]}[k]
    horizontal = k in ("w", "min-w", "max-w", "size")
    if v.kind == "num":
        return [f"{k}-{v.value}"] if is_spacing_num(v.value) else []
    if v.kind == "unit":
        return [f"{k}-[{v.value}]"] if unit_ok(v, LENGTH_UNITS) else []
    if v.kind == "pct":
        return [f"{k}-[{v.value}%]"]
    if v.kind == "frac":
        return [f"{k}-{v.value}"]
    if v.kind == "sz":
        if v.value == "full":
            return [f"{k}-full"]
        if v.value == "none":
            return ["max-w-none"] if k == "max-w" else [f"{k}-0"]
        s = shift_size(v.value, v.intensity, "xs", "7xl")
        if horizontal:
            return [f"{k}-{s}"]
        n = SZ_HEIGHT.get(s)
        return [f"{k}-{n}"] if n else []
    if v.kind == "kw":
        if v.value == "screen":
            return [f"{k}-screen"]
        if v.value in ("auto", "fit", "min", "max"):
            return [f"{k}-{v.value}"]
        if v.value == "prose":
            return ["max-w-prose"]
    if v.kind == "spec" and v.value in ("full-width", "full-height"):
        return [f"{k}-full"]
    return []


def color_class(prefix: str, v: Value) -> list[str]:
    if v.kind == "col":
        return [f"{prefix}-{v.value}"]
    if v.kind == "mod":
        return [f"{prefix}-gray-{shift_shade(v.value, v.intensity)}"]
    return []


def text_value(v: Value | None, neg: bool) -> list[str]:
    if v is None:
        return []
    if v.kind == "sz":
        if v.value in ("none", "full"):
            return []
        s = shift_size(v.value, v.intensity, "xs", "9xl")
        return [f"text-{'base' if s == 'md' else s}"]
    if v.kind == "col":
        return [f"text-{v.value}"]
    if v.kind == "mod":
        if v.value == "300":
            return ["font-light"]
        return [f"text-gray-{shift_shade(v.value, v.intensity)}"]
    if v.kind == "wt":
        return ["font-normal"] if neg else [f"font-{shift_weight(v.value, v.intensity)}"]
    if v.kind == "num":
        return [f"text-[{v.value}px]"] if is_int(v.value, 6, 200) else []
    if v.kind == "unit":
        return [f"text-[{v.value}]"] if unit_ok(v, LENGTH_UNITS) else []
    if v.kind == "spec":
        return {"center": ["text-center"], "hcenter": ["text-center"], "vcenter": ["align-middle"], "sr-only": ["sr-only"]}.get(v.value, [])
    if v.kind == "kw":
        return kw_text(v.value, neg)
    return []


def shift_weight(w: str, by: int) -> str:
    if w not in WEIGHTS or by == 0:
        return w
    return WEIGHTS[max(0, min(len(WEIGHTS) - 1, WEIGHTS.index(w) + by))]


TEXT_KW = {
    "center": "text-center", "left": "text-left", "right": "text-right", "justify": "text-justify", "start": "text-start", "end": "text-end",
    "uppercase": "uppercase", "lowercase": "lowercase", "capitalize": "capitalize", "normal-case": "normal-case",
    "italic": "italic", "not-italic": "not-italic", "underline": "underline", "no-underline": "no-underline", "line-through": "line-through", "overline": "overline",
    "truncate": "truncate", "nowrap": "text-nowrap", "wrap": "text-wrap", "balance": "text-balance", "pretty": "text-pretty", "break-words": "break-words", "break-all": "break-all",
    "mono": "font-mono", "serif": "font-serif", "sans": "font-sans", "tighter": "tracking-tighter", "tight": "tracking-tight", "wide": "tracking-wide", "wider": "tracking-wider", "widest": "tracking-widest",
    "snug": "leading-snug", "relaxed": "leading-relaxed", "loose": "leading-loose", "antialiased": "antialiased", "pre": "whitespace-pre", "pre-wrap": "whitespace-pre-wrap", "pre-line": "whitespace-pre-line",
    "hidden": "hidden", "select-none": "select-none", "select-all": "select-all", "select-text": "select-text", "invisible": "invisible", "grayscale": "grayscale",
}
NEG_KW = {"uppercase": "normal-case", "lowercase": "normal-case", "capitalize": "normal-case", "italic": "not-italic", "underline": "no-underline", "wrap": "text-nowrap", "nowrap": "text-wrap", "truncate": "text-wrap"}


def kw_text(kw: str, neg: bool) -> list[str]:
    if neg and kw in NEG_KW:
        return [NEG_KW[kw]]
    c = TEXT_KW.get(kw)
    return [c] if c else []


STANDALONE_KW = {
    "center": "flex items-center justify-center", "between": "flex justify-between", "around": "flex justify-around", "evenly": "flex justify-evenly", "stretch": "items-stretch", "baseline": "items-baseline",
    "row": "flex flex-row", "col": "flex flex-col", "row-reverse": "flex flex-row-reverse", "col-reverse": "flex flex-col-reverse", "reverse": "flex-row-reverse", "wrap": "flex-wrap", "nowrap": "whitespace-nowrap", "wrap-reverse": "flex-wrap-reverse",
    "visible": "block", "block": "block", "inline": "inline", "inline-block": "inline-block", "inline-flex": "inline-flex", "inline-grid": "inline-grid", "contents": "contents", "table": "table",
    "absolute": "absolute", "relative": "relative", "fixed": "fixed", "sticky": "sticky", "static": "static", "scroll": "overflow-scroll", "overflow-auto": "overflow-auto", "clip": "overflow-hidden", "overflow-visible": "overflow-visible",
    "pointer": "cursor-pointer", "not-allowed": "cursor-not-allowed", "wait": "cursor-wait", "grab": "cursor-grab", "move": "cursor-move", "text-cursor": "cursor-text", "default-cursor": "cursor-default",
    "events-none": "pointer-events-none", "events-auto": "pointer-events-auto", "cover": "object-cover", "contain": "object-contain", "fill": "object-fill", "square": "aspect-square", "video": "aspect-video", "aspect-auto": "aspect-auto",
    "spin": "animate-spin", "ping": "animate-ping", "pulse": "animate-pulse", "bounce": "animate-bounce", "disc": "list-disc", "decimal": "list-decimal", "first": "order-first", "last": "order-last",
    "colors": "transition-colors", "all": "transition-all", "opacity": "transition-opacity", "shadow": "transition-shadow", "transform": "transition-transform", "linear": "ease-linear", "ease-in": "ease-in", "ease-out": "ease-out", "ease-in-out": "ease-in-out",
    "fast": "duration-150", "slow": "duration-500", "grow": "grow", "no-grow": "grow-0", "shrink": "shrink", "no-shrink": "shrink-0", "flex-none": "flex-none", "flex-auto": "flex-auto", "blur": "blur-sm", "isolate": "isolate", "group": "group", "peer": "peer",
    "mx-auto": "mx-auto", "screen": "h-screen", "fit": "w-fit", "min": "w-min", "max": "w-max", "prose": "max-w-prose", "top": "top-0", "bottom": "bottom-0",
}
STANDALONE_SPEC = {
    "center": "flex items-center justify-center", "vcenter": "flex items-center", "hcenter": "flex justify-center", "fullscreen": "w-screen h-screen", "cover-parent": "absolute inset-0",
    "pin-top": "top-0", "pin-bottom": "bottom-0", "pin-left": "left-0", "pin-right": "right-0", "top-right": "top-0 right-0", "top-left": "top-0 left-0", "bottom-right": "bottom-0 right-0", "bottom-left": "bottom-0 left-0",
    "full-width": "w-full", "full-height": "h-full", "flex-center": "flex items-center justify-center", "sr-only": "sr-only",
}
NEG_STANDALONE = {"visible": "hidden", "hidden": "block", "grow": "grow-0", "shrink": "shrink-0", "scroll": "overflow-hidden", "pointer": "cursor-default", "wrap": "flex-nowrap", "nowrap": "flex-wrap", "blur": "blur-none", "center": "", "italic": "not-italic", "underline": "no-underline", "uppercase": "normal-case"}


def standalone(v: Value, neg: bool) -> list[str]:
    """Classes for a VAL with no compatible PROP in its segment."""
    if v.kind == "col":
        return ["bg-transparent"] if neg else [f"bg-{v.value}"]
    if v.kind == "mod":
        s = shift_shade(v.value, v.intensity)
        return [f"bg-gray-{s}"] + (["text-white"] if int(s) >= 700 else [])
    if v.kind == "wt":
        return ["font-normal"] if neg else [f"font-{shift_weight(v.value, v.intensity)}"]
    if v.kind == "rad":
        return ["rounded-none"] if neg else [f"rounded-{v.value}"]
    if v.kind == "sz":
        if v.value == "full":
            return ["w-full"]
        if v.value == "none":
            return []
        s = shift_size(v.value, v.intensity, "xs", "9xl")
        return [f"text-{'base' if s == 'md' else s}"]
    if v.kind == "frac":
        return [f"w-{v.value}"]
    if v.kind == "pct":
        return [f"w-[{v.value}%]"]
    if v.kind == "spec":
        return STANDALONE_SPEC.get(v.value, "").split()
    if v.kind == "lit":
        return [v.value]
    if v.kind == "kw":
        if neg:
            if v.value in NEG_STANDALONE:
                return NEG_STANDALONE[v.value].split()
            if v.value in TEXT_KW and v.value in NEG_KW:
                return [NEG_KW[v.value]]
        if v.value in STANDALONE_KW:
            return STANDALONE_KW[v.value].split()
        c = TEXT_KW.get(v.value)
        return [c] if c else []
    return []


def border_value(k: str, v: Value | None, neg: bool) -> list[str]:
    if neg:
        return [f"{k}-0"]
    if v is None:
        return [k]
    if v.kind == "col":
        return [k, f"{k}-{v.value}"]
    if v.kind == "mod":
        return [k, f"{k}-gray-{shift_shade(v.value, v.intensity)}"]
    if v.kind == "num":
        return ([k] if v.value == "1" else [f"{k}-{v.value}"]) if v.value in ("0", "1", "2", "4", "8") else []
    if v.kind == "unit":
        return [f"{k}-[{v.value}]"] if unit_ok(v, ("px",)) else []
    if v.kind == "sz":
        s = shift_size(v.value, v.intensity, "xs", "xl") if v.value in SIZES else v.value
        return {"none": [f"{k}-0"], "xs": [k], "sm": [k], "md": [f"{k}-2"], "lg": [f"{k}-4"], "xl": [f"{k}-8"], "full": [f"{k}-8"]}.get(s, [k])
    if v.kind == "kw" and v.value in ("dashed", "dotted", "solid", "double", "none"):
        return [f"{k}-0"] if v.value == "none" else [k, f"border-{v.value}"]
    return []


def rounded_value(k: str, v: Value | None, neg: bool) -> list[str]:
    if neg:
        return [f"{k}-none"]
    if v is None:
        return [f"{k}-lg"]
    if v.kind == "rad":
        return [f"{k}-{v.value}"]
    if v.kind == "sz":
        if v.value == "full":
            return [f"{k}-full"]
        if v.value == "none":
            return [f"{k}-none"]
        return [f"{k}-{shift_size(v.value, v.intensity, 'xs', '4xl')}"]
    if v.kind == "num":
        return [f"{k}-[{v.value}px]"] if is_int(v.value, 0, 64) else []
    if v.kind == "unit":
        return [f"{k}-[{v.value}]"] if unit_ok(v, LENGTH_UNITS) else []
    if v.kind == "kw" and v.value == "square":
        return [f"{k}-none"]
    return []


def ring_value(k: str, v: Value | None, neg: bool) -> list[str]:
    base = "ring-2" if k == "ring" else "outline-2"
    if neg:
        return ["ring-0"] if k == "ring" else ["outline-hidden"]
    if v is None:
        return [base]
    if v.kind == "col":
        return [base, f"{k}-{v.value}"]
    if v.kind == "mod":
        return [base, f"{k}-gray-{shift_shade(v.value, v.intensity)}"]
    if v.kind == "num":
        return [f"{k}-{v.value}"] if v.value in ("0", "1", "2", "4", "8") else []
    if v.kind == "sz":
        s = shift_size(v.value, v.intensity, "xs", "xl") if v.value in SIZES else v.value
        return {"none": ["ring-0" if k == "ring" else "outline-hidden"], "xs": [f"{k}-1"], "sm": [f"{k}-1"], "md": [f"{k}-2"], "lg": [f"{k}-4"], "xl": [f"{k}-8"], "full": [f"{k}-8"]}.get(s, [base])
    if v.kind == "kw" and v.value in ("hidden", "none"):
        return ["ring-0"] if k == "ring" else ["outline-hidden"]
    return []


def shadow_value(v: Value | None, neg: bool) -> list[str]:
    if neg:
        return ["shadow-none"]
    if v is None:
        return ["shadow-md"]
    if v.kind == "sz":
        if v.value == "none":
            return ["shadow-none"]
        if v.value == "full":
            return ["shadow-2xl"]
        s = shift_size(v.value, v.intensity, "xs", "2xl")
        return [f"shadow-{s}"]
    if v.kind == "col":
        return ["shadow-md", f"shadow-{v.value}"]
    if v.kind == "mod":
        return ["shadow-md", f"shadow-gray-{shift_shade(v.value, v.intensity)}"]
    if v.kind == "kw" and v.value in ("hidden", "none"):
        return ["shadow-none"]
    return []


def opacity_value(v: Value | None, neg: bool) -> list[str]:
    if neg:
        return ["opacity-100"]
    if v is None:
        return ["opacity-50"]
    if v.kind in ("num", "pct"):
        n = int(float(v.value) * 100) if v.kind == "num" and float(v.value) <= 1 and "." in v.value else int(float(v.value))
        return [f"opacity-{n}"] if 0 <= n <= 100 and re.fullmatch(r"\d+(\.\d+)?", v.value) else []
    if v.kind == "frac":
        a, b = v.value.split("/")
        return [f"opacity-{round(int(a) * 100 / int(b))}"]
    if v.kind == "sz":
        n = SZ_OPACITY.get(v.value)
        return [f"opacity-{n}"] if n else []
    if v.kind == "col" and v.value == "transparent":
        return ["opacity-0"]
    if v.kind == "kw" and v.value in ("hidden", "invisible"):
        return ["opacity-0"]
    return []


def z_value(v: Value | None, neg: bool) -> list[str]:
    if neg:
        return ["z-0"]
    if v is None:
        return ["z-10"]
    if v.kind == "num":
        return [f"z-{v.value}"] if is_int(v.value, 0, 100) else []
    if v.kind == "kw":
        return {"top": ["z-50"], "first": ["z-50"], "bottom": ["z-0"], "last": ["z-0"], "auto": ["z-auto"], "negative": ["-z-10"]}.get(v.value, [])
    if v.kind == "sz":
        return {"none": ["z-0"], "xs": ["z-0"], "sm": ["z-10"], "md": ["z-20"], "lg": ["z-30"], "xl": ["z-40"], "2xl": ["z-50"], "full": ["z-50"]}.get(v.value, [])
    return []


def flex_value(v: Value | None, neg: bool) -> list[str]:
    if neg:
        return ["block"]
    if v is None:
        return ["flex"]
    if v.kind == "kw":
        if v.value in ("row", "col", "row-reverse", "col-reverse", "wrap", "nowrap", "wrap-reverse"):
            return ["flex", f"flex-{v.value}"]
        if v.value in ("center",):
            return ["flex", "items-center", "justify-center"]
        if v.value in ("between", "around", "evenly", "start", "end"):
            return ["flex", f"justify-{v.value}"]
        if v.value in ("stretch", "baseline"):
            return ["flex", f"items-{v.value}"]
        if v.value == "grow":
            return ["flex-1"]
        if v.value == "auto":
            return ["flex-auto"]
        if v.value in ("inline-flex", "inline"):
            return ["inline-flex"]
        if v.value == "reverse":
            return ["flex", "flex-row-reverse"]
    if v.kind == "num":
        return ["flex-1"] if v.value == "1" else []
    if v.kind == "sz" and v.value == "none":
        return ["flex-none"]
    if v.kind == "spec":
        return {"center": ["flex", "items-center", "justify-center"], "vcenter": ["flex", "items-center"], "hcenter": ["flex", "justify-center"], "flex-center": ["flex", "items-center", "justify-center"]}.get(v.value, [])
    return []


JUSTIFY_MAP = {"start": "start", "end": "end", "center": "center", "between": "between", "around": "around", "evenly": "evenly", "stretch": "stretch", "left": "start", "right": "end"}
ITEMS_MAP = {"start": "start", "end": "end", "center": "center", "baseline": "baseline", "stretch": "stretch", "top": "start", "bottom": "end"}


def simple_kw(prefix: str, allowed: dict[str, str], v: Value | None, default: str, neg: str | None) -> list[str]:
    if v is None:
        return [f"{prefix}-{default}"]
    if v.kind == "kw" and v.value in allowed:
        return [f"{prefix}-{allowed[v.value]}"]
    if v.kind == "spec" and v.value in ("center", "hcenter", "vcenter", "flex-center"):
        return [f"{prefix}-center"]
    return []


def emit(k: str, v: Value | None, neg: bool) -> list[str]:
    """Classes for one (property, value) pair. Mirror of emit() in src/compile.ts."""
    if k.startswith("preset:"):
        return PRESETS[k[7:]].split()
    if v is not None and v.kind == "lit":
        return [v.value]
    if k in SPACING or k in INSETS:
        return spacing_value(k, v, neg)
    if k in SIZING:
        return sizing_value(k, v, neg)
    if k == "text":
        return text_value(v, neg)
    if k == "font":
        if v is None:
            return []
        if v.kind == "num" and v.value in WEIGHT_NUM:
            return [f"font-{WEIGHT_NUM[v.value]}"]
        if v.kind == "sz":
            return text_value(v, neg)
        if v.kind == "mod" and v.value == "300":
            return ["font-light"]
        if v.kind == "mod" and v.value == "600":
            return ["font-bold"]
        return text_value(v, neg)
    if k == "family":
        if v is None:
            return ["font-sans"]
        return [f"font-{v.value}"] if v.kind == "kw" and v.value in ("sans", "serif", "mono") else []
    if k == "tracking":
        if neg:
            return ["tracking-normal"]
        if v is None:
            return ["tracking-wide"]
        if v.kind == "kw" and v.value in ("tighter", "tight", "wide", "wider", "widest"):
            return [f"tracking-{v.value}"]
        if v.kind == "sz":
            return {"none": ["tracking-normal"], "xs": ["tracking-tighter"], "sm": ["tracking-tight"], "md": ["tracking-normal"], "lg": ["tracking-wide"], "xl": ["tracking-wider"], "2xl": ["tracking-widest"], "full": ["tracking-widest"]}.get(v.value, [])
        return []
    if k == "leading":
        if neg:
            return ["leading-none"]
        if v is None:
            return ["leading-relaxed"]
        if v.kind == "kw" and v.value in ("tight", "snug", "relaxed", "loose"):
            return [f"leading-{v.value}"]
        if v.kind == "sz":
            return {"none": ["leading-none"], "xs": ["leading-none"], "sm": ["leading-tight"], "md": ["leading-normal"], "lg": ["leading-relaxed"], "xl": ["leading-loose"], "2xl": ["leading-loose"]}.get(v.value, [])
        if v.kind == "num":
            return [f"leading-{v.value}"] if is_int(v.value, 3, 10) else []
        return []
    if k == "align":
        if v is None:
            return ["text-center"]
        if v.kind == "kw" and v.value in ("left", "center", "right", "justify", "start", "end"):
            return [f"text-{v.value}"]
        if v.kind == "spec" and v.value in ("center", "hcenter"):
            return ["text-center"]
        if v.kind == "kw" and v.value in ("top", "bottom", "baseline"):
            return [f"align-{v.value}"]
        if v.kind == "spec" and v.value == "vcenter":
            return ["align-middle"]
        return []
    if k == "transform":
        if neg:
            return ["normal-case"]
        if v is None:
            return []
        return kw_text(v.value, False) if v.kind == "kw" and v.value in ("uppercase", "lowercase", "capitalize", "normal-case") else []
    if k == "decoration":
        if neg:
            return ["no-underline"]
        if v is None:
            return ["underline"]
        return kw_text(v.value, False) if v.kind == "kw" and v.value in ("underline", "no-underline", "line-through", "overline") else []
    if k == "wrap":
        if neg:
            return ["text-nowrap"]
        if v is None:
            return ["text-wrap"]
        return kw_text(v.value, False) if v.kind == "kw" and v.value in ("truncate", "nowrap", "wrap", "balance", "pretty", "break-words", "break-all") else []
    if k == "whitespace":
        if v is None or (v.kind == "kw" and v.value == "wrap"):
            return ["whitespace-normal"]
        return [f"whitespace-{v.value}"] if v.kind == "kw" and v.value in ("pre", "pre-wrap", "pre-line", "nowrap") else []
    if k == "bg":
        if neg or (v is not None and v.kind == "sz" and v.value == "none"):
            return ["bg-transparent"]
        if v is None:
            return []
        return color_class("bg", v)
    if k.startswith("border") and k != "border-style":
        return border_value(k, v, neg)
    if k == "border-style":
        return [f"border-{v.value}"] if v is not None and v.kind == "kw" and v.value in ("dashed", "dotted", "solid", "double") else ["border-solid"]
    if k.startswith("rounded"):
        return rounded_value(k, v, neg)
    if k in ("ring", "outline"):
        return ring_value(k, v, neg)
    if k == "shadow":
        return shadow_value(v, neg)
    if k == "opacity":
        return opacity_value(v, neg)
    if k == "z":
        return z_value(v, neg)
    if k == "position":
        if v is None:
            return ["relative"]
        return [v.value] if v.kind == "kw" and v.value in ("absolute", "relative", "fixed", "sticky", "static") else []
    if k in ("overflow", "overflow-x", "overflow-y"):
        if neg:
            return [f"{k}-hidden"]
        if v is None:
            return [f"{k}-auto"]
        if v.kind == "kw":
            m = {"hidden": "hidden", "clip": "hidden", "scroll": "scroll", "overflow-auto": "auto", "auto": "auto", "visible": "visible", "overflow-visible": "visible"}
            return [f"{k}-{m[v.value]}"] if v.value in m else []
        return []
    if k == "display":
        if neg:
            return ["hidden"]
        if v is None:
            return ["block"]
        if v.kind == "kw":
            if v.value == "visible":
                return ["block"]
            if v.value in ("block", "inline", "inline-block", "flex", "inline-flex", "grid", "inline-grid", "hidden", "contents", "table"):
                return [v.value]
        return []
    if k == "flex":
        return flex_value(v, neg)
    if k == "direction":
        if v is None:
            return ["flex-row"]
        if v.kind == "kw" and v.value in ("row", "col", "row-reverse", "col-reverse"):
            return [f"flex-{v.value}"]
        if v.kind == "kw" and v.value == "reverse":
            return ["flex-row-reverse"]
        return []
    if k == "flexwrap":
        if neg:
            return ["flex-nowrap"]
        if v is None:
            return ["flex-wrap"]
        return [f"flex-{v.value}"] if v.kind == "kw" and v.value in ("wrap", "nowrap", "wrap-reverse") else []
    if k == "grow":
        if neg or (v is not None and ((v.kind == "num" and v.value == "0") or (v.kind == "kw" and v.value == "no-grow"))):
            return ["grow-0"]
        return ["grow"]
    if k == "shrink":
        if neg or (v is not None and ((v.kind == "num" and v.value == "0") or (v.kind == "kw" and v.value == "no-shrink"))):
            return ["shrink-0"]
        return ["shrink"]
    if k == "justify":
        return simple_kw("justify", JUSTIFY_MAP, v, "center", None)
    if k == "items":
        return simple_kw("items", ITEMS_MAP, v, "center", None)
    if k == "self":
        return simple_kw("self", {"start": "start", "end": "end", "center": "center", "stretch": "stretch", "auto": "auto", "baseline": "baseline"}, v, "center", None)
    if k == "place":
        return simple_kw("place-items", {"start": "start", "end": "end", "center": "center", "stretch": "stretch"}, v, "center", None)
    if k == "grid":
        if v is None:
            return ["grid"]
        if v.kind == "num":
            return ["grid", f"grid-cols-{v.value}"] if is_int(v.value, 1, 12) else []
        if v.kind == "spec" and v.value in ("center", "flex-center"):
            return ["grid", "place-items-center"]
        if v.kind == "kw" and v.value == "center":
            return ["grid", "place-items-center"]
        return []
    if k == "cols":
        if v is None:
            return ["grid", "grid-cols-2"]
        if v.kind == "num":
            return ["grid", f"grid-cols-{v.value}"] if is_int(v.value, 1, 12) else []
        if v.kind == "sz" and v.value == "none":
            return ["grid-cols-none"]
        return []
    if k == "rows":
        if v is None:
            return ["grid", "grid-rows-2"]
        return ["grid", f"grid-rows-{v.value}"] if v.kind == "num" and is_int(v.value, 1, 6) else []
    if k == "col-span":
        if v is None:
            return ["col-span-2"]
        if v.kind == "num":
            return [f"col-span-{v.value}"] if is_int(v.value, 1, 12) else []
        if v.kind == "sz" and v.value == "full":
            return ["col-span-full"]
        return []
    if k == "row-span":
        if v is None:
            return ["row-span-2"]
        return [f"row-span-{v.value}"] if v.kind == "num" and is_int(v.value, 1, 6) else []
    if k == "order":
        if v is None:
            return []
        if v.kind == "num":
            return [f"order-{v.value}"] if is_int(v.value, 1, 12) else []
        return [f"order-{v.value}"] if v.kind == "kw" and v.value in ("first", "last") else []
    if k == "transition":
        if neg:
            return ["transition-none"]
        if v is None:
            return ["transition"]
        if v.kind == "kw":
            if v.value in ("colors", "all", "opacity", "shadow", "transform"):
                return [f"transition-{v.value}"]
            if v.value == "fast":
                return ["transition", "duration-150"]
            if v.value == "slow":
                return ["transition", "duration-500"]
            if v.value in ("linear", "ease-in", "ease-out", "ease-in-out"):
                return ["transition", f"ease-{v.value}" if v.value == "linear" else v.value]
        if v.kind == "num":
            return ["transition", f"duration-{v.value}"] if is_int(v.value, 0, 5000) else []
        if v.kind == "unit" and v.value.endswith("ms"):
            return ["transition", f"duration-{v.value[:-2]}"]
        if v.kind == "unit" and v.value.endswith("s"):
            return ["transition", f"duration-{int(float(v.value[:-1]) * 1000)}"]
        if v.kind == "sz":
            if v.value == "none":
                return ["transition-none"]
            d = SZ_DURATION.get(shift_size(v.value, v.intensity, "xs", "2xl"))
            return ["transition", f"duration-{d}"] if d else ["transition"]
        return []
    if k in ("duration", "delay"):
        if v is None:
            return [f"{k}-300"] if k == "duration" else ["delay-150"]
        if v.kind == "num":
            return [f"{k}-{v.value}"] if is_int(v.value, 0, 5000) else []
        if v.kind == "unit" and v.value.endswith("ms"):
            return [f"{k}-{v.value[:-2]}"]
        if v.kind == "unit" and v.value.endswith("s"):
            return [f"{k}-{int(float(v.value[:-1]) * 1000)}"]
        if v.kind == "kw" and v.value == "fast":
            return [f"{k}-150"]
        if v.kind == "kw" and v.value == "slow":
            return [f"{k}-500"]
        if v.kind == "sz":
            d = SZ_DURATION.get(shift_size(v.value, v.intensity, "xs", "2xl"))
            return [f"{k}-{d}"] if d else []
        return []
    if k == "ease":
        if v is None:
            return ["ease-in-out"]
        if v.kind == "kw" and v.value == "linear":
            return ["ease-linear"]
        return [v.value] if v.kind == "kw" and v.value in ("ease-in", "ease-out", "ease-in-out") else []
    if k == "animate":
        if neg or (v is not None and v.kind == "sz" and v.value == "none"):
            return ["animate-none"]
        if v is None:
            return ["animate-pulse"]
        return [f"animate-{v.value}"] if v.kind == "kw" and v.value in ("spin", "ping", "pulse", "bounce") else []
    if k == "cursor":
        if v is None:
            return ["cursor-pointer"]
        if v.kind == "kw":
            m = {"pointer": "pointer", "not-allowed": "not-allowed", "wait": "wait", "grab": "grab", "move": "move", "text-cursor": "text", "default-cursor": "default", "auto": "auto"}
            return [f"cursor-{m[v.value]}"] if v.value in m else []
        return []
    if k == "select":
        if neg or v is None:
            return ["select-none"]
        if v.kind == "kw":
            m = {"select-none": "none", "select-all": "all", "select-text": "text", "all": "all", "text-cursor": "text"}
            return [f"select-{m[v.value]}"] if v.value in m else []
        if v.kind == "sz" and v.value == "none":
            return ["select-none"]
        return []
    if k == "pointer":
        if neg or v is None:
            return ["pointer-events-none"]
        if v.kind == "kw" and v.value in ("events-none", "events-auto", "auto"):
            return ["pointer-events-auto" if v.value != "events-none" else "pointer-events-none"]
        if v.kind == "sz" and v.value == "none":
            return ["pointer-events-none"]
        return []
    if k == "object":
        if v is None:
            return ["object-cover"]
        if v.kind == "kw" and v.value in ("cover", "contain", "fill", "center", "top", "bottom", "left", "right"):
            return [f"object-{v.value}"]
        return []
    if k == "aspect":
        if v is None:
            return ["aspect-square"]
        if v.kind == "kw" and v.value in ("square", "video"):
            return [f"aspect-{v.value}"]
        if v.kind == "kw" and v.value in ("aspect-auto", "auto"):
            return ["aspect-auto"]
        if v.kind == "frac":
            return [f"aspect-{v.value}"]
        return []
    if k == "scale":
        if neg:
            return ["scale-100"]
        if v is None:
            return ["scale-105"]
        if v.kind == "num":
            f = float(v.value)
            n = int(round(f * 100)) if f <= 3 else int(f)
            return [f"scale-{n}"] if 0 <= n <= 200 else []
        if v.kind == "pct":
            return [f"scale-{v.value}"] if is_int(v.value, 0, 200) else []
        if v.kind == "sz":
            return {"none": ["scale-100"], "xs": ["scale-95"], "sm": ["scale-105"], "md": ["scale-110"], "lg": ["scale-125"], "xl": ["scale-150"], "2xl": ["scale-150"]}.get(v.value, [])
        return []
    if k == "rotate":
        if neg:
            return ["rotate-0"]
        if v is None:
            return ["rotate-45"]
        if v.kind == "num":
            return [f"rotate-{v.value}"] if is_int(v.value, 0, 360) else []
        if v.kind == "frac" and v.value == "1/2":
            return ["rotate-180"]
        if v.kind == "frac" and v.value == "1/4":
            return ["rotate-90"]
        return []
    if k == "blur":
        if neg:
            return ["blur-none"]
        if v is None:
            return ["blur-sm"]
        if v.kind == "sz":
            if v.value == "none":
                return ["blur-none"]
            return [f"blur-{shift_size(v.value, v.intensity, 'xs', '3xl')}"]
        return []
    if k == "list":
        if neg:
            return ["list-none"]
        if v is None:
            return ["list-disc"]
        if v.kind == "kw" and v.value in ("disc", "decimal"):
            return [f"list-{v.value}"]
        if v.kind == "sz" and v.value == "none":
            return ["list-none"]
        return []
    if k == "visibility":
        if neg:
            return ["invisible"]
        if v is None:
            return ["visible"]
        if v.kind == "kw" and v.value in ("hidden", "invisible"):
            return ["invisible"]
        if v.kind == "kw" and v.value == "visible":
            return ["visible"]
        return []
    if k == "columns":
        if v is None:
            return ["columns-2"]
        return [f"columns-{v.value}"] if v.kind == "num" and is_int(v.value, 1, 12) else []
    if k == "lineclamp":
        if neg:
            return ["line-clamp-none"]
        if v is None:
            return ["line-clamp-3"]
        return [f"line-clamp-{v.value}"] if v.kind == "num" and is_int(v.value, 1, 6) else []
    if k == "container":
        return ["container", "mx-auto"]
    return []


# Which value kinds a property accepts (for pairing); mirror of ACCEPTS in compile.ts.
def accepts(k: str, v: Value) -> bool:
    return len(emit(k, v, False)) > 0


__all__ = ["Value", "parse_value", "emit", "accepts", "standalone", "PROP_OF", "VAL_OF", "MOD_OF", "HUES", "SHADES", "SIZES", "WEIGHTS", "shift_size", "shift_shade", "shift_weight"]
