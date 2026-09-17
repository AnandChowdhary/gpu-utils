"""Smoke-test scene: `cd video && .venv/bin/manim -ql example_pipeline.py ExamplePipeline`."""

from manim import DOWN, FadeIn, LaggedStart, VGroup

from style import ExplainerScene, TokenCell, activation_strip, note


class ExamplePipeline(ExplainerScene):
    def construct(self):
        cap = self.swap_caption(None, "Source text")
        cells = VGroup(*[TokenCell(t, muted=t == " ") for t in ["const", " ", "answer", " ", "=", " ", "42"]])
        cells.arrange(buff=0.08)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.08) for c in cells], lag_ratio=0.1))
        cap = self.swap_caption(cap, "Sparse features")
        strips = VGroup(*[activation_strip(c.source, c.get_center() + DOWN * 1.2) for c in cells])
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.05))
        self.play(FadeIn(note("no vocabulary, no server")))
        self.wait(1.5)
