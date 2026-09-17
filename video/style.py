"""Shared look for every gpu-utils explainer. Mirrors the conventions in
vercel-labs/gpu-lexer's video/gpu_lexer_pipeline.py: black frame, Geist Mono only,
one accent color for "learned" state, semantic colors reserved for model outputs.

Import with `from style import *` from a scene file in this directory.
"""

from __future__ import annotations

from pathlib import Path

import manimpango
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    UP,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    smooth,
)

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
for font_file in [FONT_DIR / "GeistMono-Regular.otf"]:
    if not manimpango.register_font(str(font_file.resolve())):
        raise RuntimeError(f"could not register {font_file}")

CODE_FONT = "Geist Mono"
TEXT_REFERENCE_SIZE = 48

BACKGROUND = "#000000"
WHITE_TEXT = "#F5F5F5"
MUTED = "#999999"
LINE = "#707070"
BOX_FILL = "#090909"
ACCENT = "#3B82F6"  # learned / activation state only


def mono(text: str, size: int = 24, color: str = WHITE_TEXT, **kwargs) -> Text:
    """Shape at 48px then scale, to avoid Pango glyph-advance snapping at small sizes."""
    return Text(
        text,
        font=CODE_FONT,
        font_size=TEXT_REFERENCE_SIZE,
        color=color,
        disable_ligatures=False,
        **kwargs,
    ).scale(size / TEXT_REFERENCE_SIZE)


def caption(text: str) -> Text:
    return mono(text, 28).move_to(UP * 3.5)


def note(text: str, y: float = -3.45, color: str = MUTED) -> Text:
    return mono(text, 17, color).move_to(UP * y)


def baseline_glyph(text: str, size: int, color: str) -> VGroup:
    """Prefix 'Ag' so isolated punctuation sits on a shared baseline, then drop it."""
    layout = mono(f"Ag{text}", size, color).move_to(ORIGIN)
    glyph = VGroup(*layout[2:])
    glyph.shift(LEFT * glyph.get_x())
    return glyph


def visible_char(source: str) -> str:
    return {" ": "␣", "\n": "↵", "\t": "⇥", "\r\n": "↵"}.get(source, source)


class TokenCell(VGroup):
    """Outlined pill holding one token: the recurring data-flow unit."""

    def __init__(self, source: str, muted: bool = False):
        visible = visible_char(source)
        width = 0.36 if len(visible) == 1 else 0.34 + len(visible) * 0.115
        self.box = RoundedRectangle(
            width=width,
            height=0.55,
            corner_radius=0.15,
            stroke_color=LINE,
            stroke_width=1.2,
            fill_color=BOX_FILL,
            fill_opacity=1,
        )
        self.glyph = baseline_glyph(visible, 17, MUTED if muted else WHITE_TEXT)
        self.source = source
        super().__init__(self.box, self.glyph)


def stable_values(key: str, count: int = 6) -> list[float]:
    """Deterministic pseudo-activations so renders are reproducible without model data."""
    state = 2166136261
    for character in key:
        state = ((state ^ ord(character)) * 16777619) & 0xFFFFFFFF
    values = []
    for index in range(count):
        state = (1664525 * (state ^ index) + 1013904223) & 0xFFFFFFFF
        values.append(0.34 + 0.64 * ((state >> 8) & 255) / 255)
    return values


def activation_strip(key: str, center, width: float = 0.12, height: float = 0.78, cells: int = 6) -> VGroup:
    values = stable_values(key, cells)
    gap = 0.018
    cell_height = (height - gap * (cells - 1)) / cells
    items = VGroup(
        *[
            RoundedRectangle(
                width=width,
                height=cell_height,
                corner_radius=min(0.02, cell_height / 4),
                stroke_width=0,
                fill_color=ACCENT,
                fill_opacity=value,
            )
            for value in values
        ]
    )
    items.arrange(UP, buff=gap).move_to(center)
    return items


class ExplainerScene(Scene):
    """Base scene: black background, global slowdown, smooth easing by default."""

    slowdown = 1.55

    def play(self, *animations, **kwargs):
        kwargs["run_time"] = kwargs.get("run_time", 1) * self.slowdown
        kwargs.setdefault("rate_func", smooth)
        return super().play(*animations, **kwargs)

    def wait(self, duration=1, **kwargs):
        return super().wait(duration * self.slowdown, **kwargs)

    def setup(self):
        self.camera.background_color = BACKGROUND

    def swap_caption(self, current: Text | None, text: str, *extra, run_time: float = 0.9) -> Text:
        from manim import FadeIn, ReplacementTransform

        nxt = caption(text)
        if current is None:
            self.play(FadeIn(nxt, shift=DOWN * 0.08), *extra, run_time=run_time)
        else:
            self.play(ReplacementTransform(current, nxt), *extra, run_time=run_time)
        return nxt
