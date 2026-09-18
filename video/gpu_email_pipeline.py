"""gpu-email explainer: `bash video/render.sh gpu-email draft`.

Storyboard: video/scenes/gpu-email.md. Numbers on screen come from
packages/gpu-email/MODEL_CARD.md.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ArcBetweenPoints,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    VGroup,
)

from style import ACCENT, LINE, MUTED, WHITE_TEXT, ExplainerScene, TokenCell, activation_strip, mono, note

EMAIL_LINES = [
    ("Hi Bob,", "greeting"),
    ("Thanks for the update, that works for me.", "reply"),
    ("Best,", "closing"),
    ("John Doe", "signature"),
    ("CEO, Acme Inc", "signature"),
    ("+1 (555) 123-4567", "signature"),
    ("john@acme.com", "signature"),
    ("On Mon, Jan 5, 2024 at 3:14 PM Bob <bob@x.com> wrote:", "attribution"),
    ("> Hi John,", "quote"),
    ("> Can we move the meeting?", "quote"),
    ("> Bob", "quote"),
]
KIND_COLOR = {
    "reply": WHITE_TEXT,
    "greeting": "#A78BFA",
    "closing": "#A78BFA",
    "signature": "#22C55E",
    "attribution": "#F59E0B",
    "quote": "#6B7280",
}
PARAMS = "170,519"
SIZE = "109 KiB"
LATENCY = "39 ms"


def email_block(tinted: bool = False) -> VGroup:
    rows = VGroup()
    for text, kind in EMAIL_LINES:
        color = KIND_COLOR[kind] if tinted else (MUTED if kind == "quote" else WHITE_TEXT)
        rows.add(mono(text, 17, color))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    return rows


class GpuEmailPipeline(ExplainerScene):
    def construct(self):
        # Scene 1: the input
        cap = self.swap_caption(None, "One email, three messages glued together")
        block = email_block().move_to(LEFT * 2.2)
        self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.08) for r in block], lag_ratio=0.08), run_time=2.2)
        self.wait(1.2)

        # Scene 2: tokens
        cap = self.swap_caption(cap, "Split into character-class runs")
        sample = ["Best", ",", "\n", "John", " ", "Doe", "\n", ">", " ", "Can", " ", "we", " ", "move", " ", "it", "?"]
        cells = VGroup(*[TokenCell(t, muted=t in (" ", "\n")) for t in sample]).arrange(buff=0.08)
        cells.move_to(DOWN * 0.4)
        keep = VGroup(block[2], block[3], block[9])
        self.play(FadeOut(VGroup(*[r for r in block if r not in keep])), run_time=0.7)
        self.play(ReplacementTransform(keep, cells), run_time=1.4)
        foot = note("no vocabulary, no server")
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(1.2)

        # Scene 3: sparse features
        cap = self.swap_caption(cap, "37 hashed feature ids per token", FadeOut(foot))
        strips = VGroup(*[activation_strip(c.source + str(i), c.get_center() + DOWN * 1.15) for i, c in enumerate(cells)])
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.04), run_time=1.4)
        groups = VGroup(
            mono("word · shape · suffix", 16, MUTED),
            mono("line: first word · ends with ':' · looks like a phone", 16, MUTED),
            mono("document: quote lines above · lines since 'wrote:'", 16, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(UP * 1.6)
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.1) for g in groups], lag_ratio=0.3), run_time=1.6)
        self.wait(1.4)

        # Scene 4: dilated convolutions
        cap = self.swap_caption(cap, "Six dilated convolutions, kernel 3", FadeOut(groups))
        arcs = VGroup()
        centers = [c.get_center() for c in cells]
        for row, dil in enumerate([1, 2, 4, 8]):
            y = 0.55 + row * 0.42
            for i in range(0, len(cells), max(1, dil)):
                j = i + dil
                if j >= len(cells):
                    continue
                a = centers[i] + UP * y
                b = centers[j] + UP * y
                arcs.add(ArcBetweenPoints(a, b, angle=-1.2, stroke_color=ACCENT, stroke_width=1.6, stroke_opacity=0.85))
        self.play(LaggedStart(*[FadeIn(a) for a in arcs], lag_ratio=0.02), run_time=2.4)
        foot = note(f"dilations 1 · 2 · 4 · 8 · 16 · 32 → 127-token field · {PARAMS} parameters · int6")
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(1.4)

        # Scene 5: two heads
        cap = self.swap_caption(cap, "Head 1 paints lines, head 2 tags contact fields", FadeOut(arcs), FadeOut(strips), FadeOut(foot))
        tinted = email_block(tinted=True).move_to(LEFT * 2.2)
        self.play(ReplacementTransform(cells, tinted), run_time=1.5)
        underlines = VGroup()
        labels = VGroup()
        for idx, field in [(3, "NAME"), (4, "TITLE"), (5, "PHONE"), (6, "EMAIL")]:
            row = tinted[idx]
            ln = Line(row.get_corner(DOWN + LEFT), row.get_corner(DOWN + RIGHT), stroke_color=KIND_COLOR["signature"], stroke_width=2).shift(DOWN * 0.03)
            underlines.add(ln)
            labels.add(mono(field, 13, KIND_COLOR["signature"]).next_to(row, RIGHT, buff=0.35))
        self.play(LaggedStart(*[FadeIn(u) for u in underlines], lag_ratio=0.2), LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2), run_time=1.6)
        self.wait(1.2)

        # Scene 6: rules + Viterbi
        cap = self.swap_caption(cap, "Exact rules first, then Viterbi over line kinds")
        rule_box = Rectangle(width=tinted[9].width + 0.3, height=1.1, stroke_color=LINE, stroke_width=1.2).move_to(VGroup(tinted[8], tinted[9], tinted[10]).get_center())
        rule_note = mono("prefix '>' → quote (rule)", 14, MUTED).next_to(rule_box, RIGHT, buff=0.4)
        self.play(FadeIn(rule_box), FadeIn(rule_note), run_time=0.9)
        arrow = Line(tinted[7].get_right() + RIGHT * 0.2, tinted[8].get_right() + RIGHT * 0.2, stroke_color=KIND_COLOR["attribution"], stroke_width=2)
        arrow_note = mono("attribution → quote: free · quote → reply: costly", 14, MUTED).next_to(rule_note, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeIn(arrow), FadeIn(arrow_note), run_time=0.9)
        self.wait(1.4)

        # Scene 7: typed output
        cap = self.swap_caption(cap, "reply · segments · contact", FadeOut(rule_box), FadeOut(rule_note), FadeOut(arrow), FadeOut(arrow_note), FadeOut(underlines), FadeOut(labels))
        out = VGroup(
            mono('reply: "Hi Bob, Thanks for the update…"', 15, WHITE_TEXT),
            mono("segments: greeting · reply · closing", 15, WHITE_TEXT),
            mono("          signature · attribution · quote", 15, WHITE_TEXT),
            mono('contact: { name: "John Doe", title: "CEO",', 15, KIND_COLOR["signature"]),
            mono('           phone: ["+1 (555) 123-4567"],', 15, KIND_COLOR["signature"]),
            mono('           email: ["john@acme.com"] }', 15, KIND_COLOR["signature"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(RIGHT * 3.3)
        self.play(tinted.animate.scale(0.78).move_to(LEFT * 3.6), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(o, shift=RIGHT * 0.1) for o in out], lag_ratio=0.15), run_time=1.6)
        foot = note(f"{SIZE} Brotli · warm CPU call {LATENCY} on a 1 KB email · probabilistic tagger + exact rules")
        self.play(FadeIn(foot), run_time=0.6)
        self.wait(2.6)
