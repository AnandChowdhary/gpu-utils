"""gpu-view explainer. `bash video/render.sh gpu-view draft`.

Storyboard: video/scenes/gpu-view.md. Every role shown here is what the shipped model
predicts for the phrase (checked against packages/gpu-view/test/parse.test.ts), and the
parameter count comes from MODEL_CARD.md.
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    ReplacementTransform,
    VGroup,
)

from style import ACCENT, MUTED, ExplainerScene, TokenCell, activation_strip, mono, note

PHRASE = "total revenue by region this quarter, top 10, as a bar chart"
TOKENS = ["total", "revenue", "by", "region", "this", "quarter", ",", "top", "10", ",", "as", "a", "bar", "chart"]
ROLES = {
    "total": ("AGG_FN", "#22C55E"),
    "revenue": ("AGG_FIELD", "#22C55E"),
    "region": ("GROUP_FIELD", "#A855F7"),
    "this": ("TIME_VALUE", "#F59E0B"),
    "quarter": ("TIME_VALUE", "#F59E0B"),
    "top": ("SORT_DIR", "#EC4899"),
    "10": ("LIMIT", "#EC4899"),
    "bar": ("CHART", "#14B8A6"),
    "chart": ("CHART", "#14B8A6"),
}
BOUNDARIES = {0, 3, 4, 7, 12}
SCHEMA_TAGS = {"revenue": "field:number", "region": "field:enum", "quarter": "time", "this": "time",
               "total": "lexicon", "top": "lexicon", "chart": "lexicon", "10": "digits"}
PARAMS = "37,775 parameters, int6"


class GpuViewPipeline(ExplainerScene):
    def construct(self):
        # Scene 1: phrase with spaces.
        cap = self.swap_caption(None, "A phrase in a search bar")
        pieces = []
        for i, t in enumerate(TOKENS):
            if i > 0 and t not in (",",):
                pieces.append(" ")
            pieces.append(t)
        cells = VGroup(*[TokenCell(p, muted=(p == " ")) for p in pieces]).arrange(buff=0.06).scale(0.82)
        cells.move_to(UP * 0.8)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.08) for c in cells], lag_ratio=0.05))
        schema = note("schema: revenue:number  region:enum  closed_at:date", y=-1.2)
        self.play(FadeIn(schema))
        self.wait(1.2)

        # Scene 2: drop whitespace.
        cap = self.swap_caption(cap, "Character-class tokens, whitespace dropped")
        words = VGroup(*[c for c in cells if c.source != " "])
        spaces = VGroup(*[c for c in cells if c.source == " "])
        target = VGroup(*[TokenCell(c.source) for c in words]).arrange(buff=0.08).scale(0.82).move_to(UP * 0.8)
        self.play(FadeOut(spaces), *[ReplacementTransform(a, b) for a, b in zip(words, target)])
        cells = target
        self.wait(0.8)

        # Scene 3: schema as anonymous features.
        cap = self.swap_caption(cap, "The schema enters as anonymous features")
        tags = VGroup()
        for cell in cells:
            tag = SCHEMA_TAGS.get(cell.source)
            if tag:
                tags.add(mono(tag, 9, MUTED).next_to(cell, DOWN, buff=0.12))
        strike = Line(schema.get_left(), schema.get_right(), color=MUTED, stroke_width=2)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.05) for t in tags], lag_ratio=0.08), Create(strike))
        strips = VGroup(*[activation_strip(c.source, c.get_center() + DOWN * 1.35, height=0.6) for c in cells])
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.04))
        foot = note("no vocabulary, no field names, no server")
        self.play(FadeOut(schema), FadeOut(strike), FadeIn(foot))
        self.wait(1.5)

        # Scene 4: bidirectional scan.
        cap = self.swap_caption(cap, "h[t] = a[t]·h[t−1] + (1−a[t])·u[t], forward and backward", FadeOut(foot))
        sweep = mono("▮", 20, ACCENT).move_to(strips[0].get_center() + LEFT * 0.4)
        self.play(FadeIn(sweep))
        self.play(sweep.animate.move_to(strips[-1].get_center() + RIGHT * 0.4), run_time=2.2)
        brighter = VGroup(*[activation_strip(c.source + "f", c.get_center() + DOWN * 1.35, height=0.6) for c in cells])
        self.play(sweep.animate.move_to(strips[0].get_center() + LEFT * 0.4), ReplacementTransform(strips, brighter), run_time=2.2)
        strips = brighter
        foot = note(PARAMS)
        self.play(FadeOut(sweep), FadeIn(foot))
        self.wait(1)

        # Scene 5: roles and boundaries.
        cap = self.swap_caption(cap, "One role per token, plus clause boundaries", FadeOut(foot), FadeOut(tags))
        labels = VGroup()
        ticks = VGroup()
        for i, cell in enumerate(cells):
            role, color = ROLES.get(cell.source, ("O", MUTED))
            labels.add(mono(role, 10, color).next_to(cell, DOWN, buff=0.12))
            if i in BOUNDARIES:
                x = cell.get_left()[0] - 0.05
                ticks.add(Line(UP * 1.15, UP * 0.45, color=ACCENT, stroke_width=2).move_to([x, 0.8, 0]))
        self.play(ReplacementTransform(strips, labels))
        self.play(LaggedStart(*[Create(t) for t in ticks], lag_ratio=0.15))
        self.wait(1.2)

        # Scene 6: compiler.
        cap = self.swap_caption(cap, "Ordinary TypeScript compiles the roles", FadeOut(ticks))
        lines = [
            ("aggregate: sum(revenue)", "#22C55E"),
            ("groupBy: region", "#A855F7"),
            ("filter: closed_at between 2026-07-01 … 2026-09-30", "#F59E0B"),
            ("limit: 10", "#EC4899"),
            ("chart: bar", "#14B8A6"),
        ]
        spec = VGroup(*[mono(t, 18, c) for t, c in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(DOWN * 1.4)
        groups = {
            "#22C55E": VGroup(*[l for l, c in zip(labels, cells) if ROLES.get(c.source, ("", ""))[1] == "#22C55E"]),
            "#A855F7": VGroup(*[l for l, c in zip(labels, cells) if ROLES.get(c.source, ("", ""))[1] == "#A855F7"]),
            "#F59E0B": VGroup(*[l for l, c in zip(labels, cells) if ROLES.get(c.source, ("", ""))[1] == "#F59E0B"]),
            "#EC4899": VGroup(*[l for l, c in zip(labels, cells) if ROLES.get(c.source, ("", ""))[1] == "#EC4899"]),
            "#14B8A6": VGroup(*[l for l, c in zip(labels, cells) if ROLES.get(c.source, ("", ""))[1] == "#14B8A6"]),
        }
        anims = []
        for line, (_t, color) in zip(spec, lines):
            anims.append(ReplacementTransform(groups[color].copy(), line))
        self.play(LaggedStart(*anims, lag_ratio=0.2), run_time=2.5)
        foot = note("unresolved words become diagnostics, never guesses")
        self.play(FadeIn(foot))
        self.wait(1.5)

        # Scene 7: hold.
        cap = self.swap_caption(cap, "gpu-view · runs in your browser", FadeOut(foot))
        self.wait(2.5)
