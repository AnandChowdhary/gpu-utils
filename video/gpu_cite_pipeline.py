"""gpu-cite explainer: reference string → tokens → sparse features → two scans → tags → record.

Render: `bash video/render.sh gpu-cite draft`. Numbers below come from packages/gpu-cite/MODEL_CARD.md.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    VGroup,
    linear,
)

from style import ACCENT, BOX_FILL, LINE, MUTED, WHITE_TEXT, ExplainerScene, TokenCell, activation_strip, mono, note

# From MODEL_CARD.md
PARAMS = "76,747 parameters"
SIZE = "52.3 KiB Brotli"
LATENCY = "~4 ms per reference, CPU path"

REFERENCE = "Smith, J., & Doe, A. (2019). A study of things. Journal of Stuff, 12(3), 45–67. doi:10.1000/xyz123"

# Semantic colours: model outputs only.
ROLE_COLORS = {
    "AUTHOR": "#F59E0B",
    "TITLE": "#10B981",
    "CONTAINER": "#8B5CF6",
    "YEAR": "#EC4899",
    "VOLUME": "#EC4899",
    "ISSUE": "#EC4899",
    "PAGES": "#EC4899",
    "DOI": "#06B6D4",
}

# (token, role) pairs for a shortened version of the reference so cells fit one row.
TOKENS = [
    ("Smith", "AUTHOR"), (",", "AUTHOR"), (" ", "AUTHOR"), ("J", "AUTHOR"), (".", "AUTHOR"), (",", None), (" ", None),
    ("&", None), (" ", None), ("Doe", "AUTHOR"), (",", "AUTHOR"), (" ", "AUTHOR"), ("A", "AUTHOR"), (".", "AUTHOR"),
    (" ", None), ("(", None), ("2019", "YEAR"), (")", None), (".", None), (" ", None),
    ("A", "TITLE"), (" ", "TITLE"), ("study", "TITLE"), (" ", "TITLE"), ("of", "TITLE"), (" ", "TITLE"), ("things", "TITLE"),
    (".", None), (" ", None), ("Journal", "CONTAINER"), (" ", "CONTAINER"), ("of", "CONTAINER"), (" ", "CONTAINER"),
    ("Stuff", "CONTAINER"), (",", None), (" ", None), ("12", "VOLUME"), ("(", None), ("3", "ISSUE"), (")", None),
    (",", None), (" ", None), ("45", "PAGES"), ("–", "PAGES"), ("67", "PAGES"), (".", None),
]


class GpuCitePipeline(ExplainerScene):
    def construct(self):
        # Scene 1: a reference
        cap = self.swap_caption(None, "A reference, any style")
        ref = mono(REFERENCE, 17).move_to(UP * 0.3)
        self.play(FadeIn(ref, shift=DOWN * 0.08))
        foot = note("APA · MLA · IEEE · Vancouver · Harvard · Nature · messy copy-paste")
        self.play(FadeIn(foot))
        self.wait(2.2)

        # Scene 2: tokens
        cap = self.swap_caption(cap, "Split into character-class tokens", FadeOut(foot))
        cells = VGroup(*[TokenCell(t, muted=(t == " ")) for t, _ in TOKENS])
        cells.arrange(buff=0.05).scale(0.62).move_to(UP * 0.3)
        self.play(ReplacementTransform(ref, cells), run_time=1.4)
        foot = note("no vocabulary, no learned tokenizer")
        self.play(LaggedStart(*[c.box.animate.set_stroke(color=WHITE_TEXT) for c in cells], lag_ratio=0.02), FadeIn(foot))
        self.wait(1.6)

        # Scene 3: sparse features
        cap = self.swap_caption(cap, "12 sparse feature ids per token", FadeOut(foot))
        strips = VGroup(*[activation_strip(f"{i}:{c.source}", c.get_center() + DOWN * 1.15, width=0.09, height=0.6) for i, c in enumerate(cells)])
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.05) for s in strips], lag_ratio=0.02), run_time=1.6)
        labels = VGroup(
            mono("word·skeleton·shape", 12, MUTED).next_to(strips[0], DOWN, buff=0.15).shift(RIGHT * 0.6),
            mono("flag: YEARLIKE", 12, MUTED).next_to(strips[16], DOWN, buff=0.15),
            mono("flag: CUE_PP / DASH", 12, MUTED).next_to(strips[43], DOWN, buff=0.15).shift(LEFT * 0.4),
        )
        self.play(FadeIn(labels))
        self.wait(2.0)

        # Scene 4: two scans
        cap = self.swap_caption(cap, "Bidirectional gated scans", FadeOut(labels))
        window = Rectangle(width=1.1, height=0.55, stroke_color=ACCENT, stroke_width=2.4, fill_color=ACCENT, fill_opacity=0.12)
        window.move_to(cells[0].get_center()).scale(0.62 / 0.62)
        self.play(FadeIn(window))
        self.play(window.animate.move_to(cells[-1].get_center()), run_time=2.4, rate_func=linear)
        self.play(window.animate.move_to(cells[0].get_center()), run_time=2.4, rate_func=linear)
        pooled = Rectangle(width=cells.width + 0.2, height=0.06, stroke_width=0, fill_color=ACCENT, fill_opacity=0.9)
        pooled.next_to(cells, UP, buff=0.18)
        foot = note("h = a·h_prev + (1−a)·b — a parallel prefix scan on the GPU")
        self.play(FadeOut(window), Create(pooled), FadeIn(foot))
        self.wait(1.2)

        # Scene 5: tags
        cap = self.swap_caption(cap, "One tag per token", FadeOut(foot), FadeOut(strips), FadeOut(pooled))
        recolor = []
        for cell, (_, role) in zip(cells, TOKENS, strict=True):
            if role:
                colour = ROLE_COLORS[role]
                recolor.append(cell.box.animate.set_stroke(color=colour).set_fill(color=colour, opacity=0.18))
            else:
                recolor.append(cell.box.animate.set_stroke(color=LINE))
        self.play(LaggedStart(*recolor, lag_ratio=0.02), run_time=1.8)
        legend = VGroup(*[mono(name, 13, colour) for name, colour in [("AUTHOR", ROLE_COLORS["AUTHOR"]), ("YEAR", ROLE_COLORS["YEAR"]), ("TITLE", ROLE_COLORS["TITLE"]), ("CONTAINER", ROLE_COLORS["CONTAINER"]), ("VOLUME·ISSUE·PAGES", ROLE_COLORS["PAGES"])]])
        legend.arrange(buff=0.5).next_to(cells, DOWN, buff=0.6)
        pill = RoundedRectangle(width=2.3, height=0.5, corner_radius=0.2, stroke_color=LINE, fill_color=BOX_FILL, fill_opacity=1)
        pill_text = mono("type: article", 15).move_to(pill)
        pill_group = VGroup(pill, pill_text).next_to(legend, DOWN, buff=0.4)
        foot = note("probabilistic tags; CRF + Viterbi on the CPU", y=-3.45)
        self.play(FadeIn(legend), FadeIn(pill_group), FadeIn(foot))
        self.wait(2.0)

        # Scene 6: compiler
        cap = self.swap_caption(cap, "Deterministic compiler", FadeOut(legend), FadeOut(pill_group), FadeOut(foot))
        lines = [
            ("authors", '[{given: "J.", family: "Smith"}, {given: "A.", family: "Doe"}]', ROLE_COLORS["AUTHOR"]),
            ("year", "2019", ROLE_COLORS["YEAR"]),
            ("title", '"A study of things"', ROLE_COLORS["TITLE"]),
            ("container", '"Journal of Stuff"', ROLE_COLORS["CONTAINER"]),
            ("volume · issue", '"12" · "3"', ROLE_COLORS["VOLUME"]),
            ("pages", '{from: "45", to: "67"}', ROLE_COLORS["PAGES"]),
            ("doi", '"10.1000/xyz123"  ← regex', ROLE_COLORS["DOI"]),
        ]
        rows = VGroup()
        for key, value, colour in lines:
            k = mono(f"{key}:", 16, colour)
            v = mono(value, 16, WHITE_TEXT)
            row = VGroup(k, v).arrange(buff=0.3, aligned_edge=UP)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(UP * 0.1)
        self.play(ReplacementTransform(cells, rows), run_time=1.6)
        foot = note("DOI, arXiv id and URL come from regex, never from the model")
        self.play(FadeIn(foot))
        self.wait(2.4)

        # Scene 7: on device
        cap = self.swap_caption(cap, "Runs on your GPU. No server.", FadeOut(foot))
        stats = VGroup(mono(PARAMS, 24), mono(SIZE, 24), mono(LATENCY, 24, MUTED)).arrange(DOWN, buff=0.35)
        stats.move_to(DOWN * 2.2)
        self.play(rows.animate.shift(UP * 0.7), FadeIn(stats, shift=UP * 0.1))
        self.wait(3.0)
