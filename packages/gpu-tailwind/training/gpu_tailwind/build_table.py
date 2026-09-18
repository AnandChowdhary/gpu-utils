"""Compile src/table.json from the Tailwind v4 default theme plus the NL lexicon.

    uv run python -m gpu_tailwind.build_table

The theme is read from training/data/tailwind-theme.txt (vendored from the
`tailwindcss` npm package, MIT); pass --fetch to re-download it from unpkg.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from .lexicon import NEG_WORDS, PRESETS, PROPS, SEP_WORDS, SPELLING, VALUES, VAR_RULES, words_of

ROOT = Path(__file__).resolve().parents[2]
THEME = ROOT / "training" / "data" / "tailwind-theme.txt"
OUT = ROOT / "src" / "table.json"
URL = "https://unpkg.com/tailwindcss@4/theme.css"


def parse_theme(css: str) -> dict[str, list[str]]:
    vars_ = dict(re.findall(r"--([a-z0-9-]+):\s*([^;]+);", css))
    hues: list[str] = []
    for name in vars_:
        m = re.match(r"color-([a-z]+)-50$", name)
        if m:
            hues.append(m.group(1))
    shades = sorted({n.rsplit("-", 1)[1] for n in vars_ if re.match(r"color-[a-z]+-\d+$", n)}, key=int)

    def scale(prefix: str, drop: tuple[str, ...] = ()) -> list[str]:
        out = []
        for n in vars_:
            if n.startswith(prefix) and "--" not in n[len(prefix) :]:
                v = n[len(prefix) :]
                if v and v not in drop:
                    out.append(v)
        return out

    return {
        "hues": hues,
        "shades": shades,
        "breakpoints": scale("breakpoint-"),
        "containers": scale("container-"),
        "textSizes": scale("text-"),
        "weights": scale("font-weight-"),
        "tracking": scale("tracking-"),
        "leading": scale("leading-"),
        "radii": scale("radius-"),
        "shadows": scale("shadow-"),
        "blurs": scale("blur-"),
        "eases": scale("ease-"),
        "animations": scale("animate-"),
    }


def main() -> None:
    if "--fetch" in sys.argv or not THEME.exists():
        THEME.parent.mkdir(parents=True, exist_ok=True)
        THEME.write_bytes(urllib.request.urlopen(URL, timeout=30).read())  # noqa: S310
    theme = parse_theme(THEME.read_text())
    norm = lambda p: " ".join(words_of(p))  # noqa: E731
    props = {norm(p): k for k, ps in PROPS.items() for p in ps}
    vals: dict[str, str] = {}
    mods: dict[str, str] = {}
    for k, ps in VALUES.items():
        for p in ps:
            if k.startswith("mod:"):
                mods.setdefault(norm(p), k[4:])
            else:
                vals.setdefault(norm(p), k)
                if re.search(r"[^a-z0-9 ]", p):
                    vals.setdefault(p.replace(" ", ""), k)
    table = {
        "theme": theme,
        "props": props,
        "vals": vals,
        "mods": mods,
        "vars": VAR_RULES,
        "presets": PRESETS,
        "seps": SEP_WORDS,
        "neg": NEG_WORDS,
        "spelling": SPELLING,
    }
    OUT.write_text(json.dumps(table, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes): {len(props)} props, {len(vals)} vals, {len(theme['hues'])} hues")


if __name__ == "__main__":
    main()
