"""Explainer for gpu-paste. `bash video/render.sh gpu-paste draft`."""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    LaggedStart,
    Rectangle,
    ReplacementTransform,
    VGroup,
)

from style import ACCENT, MUTED, WHITE_TEXT, ExplainerScene, TokenCell, activation_strip, mono, note

PERSON = "#F59E0B"
COMPANY = "#10B981"
PHONE = "#EC4899"

# Real numbers from MODEL_CARD.md.
PARAMS = "68,659 parameters"
SIZE = "55 KB Brotli"

LINES = [
    ["Jane", " ", "Doe"],
    ["Acme", " ", "Corp"],
    ["+", "1", " ", "555", " ", "0100"],
    ["jane", "@", "acme", ".", "com"],
]


def row(tokens: list[str]) -> VGroup:
    cells = VGroup(*[TokenCell(t, muted=t == " ") for t in tokens])
    cells.arrange(buff=0.06)
    return cells


class GpuPastePipeline(ExplainerScene):
    def construct(self):
        cap = self.swap_caption(None, "Pasted text")
        rows = VGroup(*[row(t) for t in LINES]).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rows.move_to(UP * 0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.08) for r in rows], lag_ratio=0.15))
        self.wait(2.0)

        # Scene 2: rules claim the email
        cap = self.swap_caption(cap, "Rules first: validated, never guessed")
        email = rows[3]
        tag = mono("email · rule", 15, WHITE_TEXT).next_to(email, RIGHT, buff=0.35)
        box = Rectangle(width=email.width + 0.16, height=email.height + 0.16, stroke_color=WHITE_TEXT, stroke_width=1.2).move_to(email)
        self.play(FadeIn(box), FadeIn(tag, shift=LEFT * 0.08))
        foot = note("JSON · CSV · URL · UUID · IP · color · money · dates → rules")
        self.play(FadeIn(foot))
        self.wait(1.6)

        # Scene 3: sparse features on the remaining cells
        cap = self.swap_caption(cap, "Each token → 10 hashed features", FadeOut(foot))
        strips = VGroup()
        for r in rows[:3]:
            for c in r:
                if c.source != " ":
                    strips.add(activation_strip(c.source, c.get_center() + DOWN * 0.62, height=0.5, width=0.1))
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.03))
        foot = note("no vocabulary: word, skeleton, shape, affix, length, position hashes")
        self.play(FadeIn(foot))
        self.wait(2.2)

        # Scene 4: bidirectional scan
        cap = self.swap_caption(cap, "Gated affine scan, both directions", FadeOut(foot), FadeOut(strips))
        left = rows.get_left()[0]
        right = rows.get_right()[0]
        bar = Rectangle(width=0.18, height=rows.height + 0.3, stroke_width=0, fill_color=ACCENT, fill_opacity=0.35)
        bar.move_to([left, rows.get_center()[1], 0])
        self.play(FadeIn(bar))
        self.play(bar.animate.move_to([right, rows.get_center()[1], 0]), run_time=1.4)
        self.play(bar.animate.move_to([left, rows.get_center()[1], 0]), run_time=1.4)
        self.play(FadeOut(bar))

        # Scene 5: span head
        cap = self.swap_caption(cap, "Span head: person · company · phone")
        colors = [PERSON, COMPANY, PHONE]
        labels = ["person", "company", "phone"]
        anims = []
        tags = VGroup()
        for r, color, label in zip(rows[:3], colors, labels):
            for c in r:
                if c.source != " ":
                    anims.append(c.box.animate.set_stroke(color, width=1.8))
                    anims.append(c.glyph.animate.set_color(color))
            t = mono(label, 15, color).next_to(r, RIGHT, buff=0.35)
            tags.add(t)
        self.play(LaggedStart(*anims, lag_ratio=0.02), FadeIn(tags, shift=LEFT * 0.08))
        self.wait(1.4)

        # Scene 6: kind head
        cap = self.swap_caption(cap, "Mean-pool → kind")
        pooled = mono("contact  0.99", 22, ACCENT).move_to(DOWN * 2.1)
        bracket = Rectangle(width=rows.width + 0.4, height=0.04, stroke_width=0, fill_color=ACCENT, fill_opacity=0.8)
        bracket.next_to(rows, DOWN, buff=0.35)
        self.play(FadeIn(bracket))
        self.play(ReplacementTransform(bracket, pooled))
        self.wait(1.2)

        # Scene 7: typed output
        cap = self.swap_caption(cap, "parse(text) →", FadeOut(pooled), FadeOut(tags), FadeOut(tag), FadeOut(box))
        out_lines = [
            ('{ kind: "contact",', WHITE_TEXT),
            ('  name: "Jane Doe",', PERSON),
            ('  company: "Acme Corp",', COMPANY),
            ('  phone: "+15550100",', PHONE),
            ('  email: "jane@acme.com" }', WHITE_TEXT),
        ]
        out = VGroup(*[mono(t, 20, c) for t, c in out_lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        out.move_to(UP * 0.4)
        self.play(ReplacementTransform(rows, out), run_time=1.3)
        self.wait(1.2)

        # Scene 8: hold with real numbers
        foot = note(f"{PARAMS} · int6 · {SIZE} · runs in your browser, no server")
        self.play(FadeIn(foot))
        self.wait(3.0)
