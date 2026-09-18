"""gpu-tailwind explainer. `bash video/render.sh gpu-tailwind draft`.

Storyboard: video/scenes/gpu-tailwind.md. Numbers come from packages/gpu-tailwind/MODEL_CARD.md.
"""

from manim import DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, LaggedStart, Line, Rectangle, ReplacementTransform, VGroup

from style import ACCENT, MUTED, ExplainerScene, TokenCell, activation_strip, mono, note

PARAMS = "45,218 parameters"
SIZE = "~56 KB Brotli"

PROPERTY = "#60A5FA"
VALUE = "#34D399"
VARIANT = "#FBBF24"

WORDS = ["card", " ", "with", " ", "rounded", " ", "corners", ",", " ", "subtle", " ", "shadow", ",", " ", "blue", " ", "on", " ", "hover", ",", " ", "hidden", " ", "on", " ", "mobile"]
ROLES = {
    "card": PROPERTY,
    "rounded": PROPERTY,
    "corners": PROPERTY,
    "subtle": VALUE,
    "shadow": PROPERTY,
    "blue": VALUE,
    "on": VARIANT,
    "hover": VARIANT,
    "hidden": VALUE,
    "mobile": VARIANT,
}
SEGMENTS = [(0, 0, "bg-white p-4"), (4, 6, "rounded-lg"), (9, 11, "shadow-sm"), (14, 18, "hover:bg-blue-500"), (21, 25, "max-sm:hidden")]


class GpuTailwindPipeline(ExplainerScene):
    def construct(self):
        # 1. phrase
        cap = self.swap_caption(None, "Natural language")
        cells = VGroup(*[TokenCell(w, muted=(w == " " or w == ",")) for w in WORDS])
        cells.arrange(buff=0.05).scale(0.82).move_to(UP * 0.9)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.08) for c in cells], lag_ratio=0.04))
        self.wait(1.2)

        # 2. features
        cap = self.swap_caption(cap, "7 hashed features per token, no vocabulary")
        strips = VGroup(*[activation_strip(c.source + str(i), c.get_center() + DOWN * 1.15, width=0.09, height=0.62) for i, c in enumerate(cells)])
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.03))
        foot = note("word · consonant skeleton · prefix · suffix · shape · length · class")
        self.play(FadeIn(foot))
        self.wait(1.5)

        # 3. scan
        cap = self.swap_caption(cap, f"Two gated affine scans, {PARAMS}", FadeOut(foot))
        window = Rectangle(width=0.9, height=0.85, stroke_color=ACCENT, stroke_width=2, fill_color=ACCENT, fill_opacity=0.12)
        window.move_to(strips[0].get_center())
        self.play(FadeIn(window), run_time=0.4)
        self.play(window.animate.move_to(strips[-1].get_center()), run_time=2.2)
        self.play(window.animate.move_to(strips[0].get_center()), run_time=2.2)
        foot = note("int6 weights · runs as WGSL compute passes")
        self.play(FadeOut(window), FadeIn(foot), run_time=0.6)
        self.wait(0.8)

        # 4. roles
        cap = self.swap_caption(cap, "Each token gets a role", FadeOut(foot), FadeOut(strips))
        recolor = []
        for c in cells:
            color = ROLES.get(c.source)
            if color:
                recolor.append(c.glyph.animate.set_color(color))
                recolor.append(c.box.animate.set_stroke(color))
        self.play(LaggedStart(*recolor, lag_ratio=0.05), run_time=1.6)
        legend = VGroup(mono("PROPERTY", 17, PROPERTY), mono("VALUE", 17, VALUE), mono("VARIANT", 17, VARIANT), mono("SEP / O", 17, MUTED)).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.4)
        self.play(FadeIn(legend))
        self.wait(1.4)

        # 5. segments
        cap = self.swap_caption(cap, "Split into segments", FadeOut(legend))
        brackets = VGroup()
        for start, end, _ in SEGMENTS:
            left = cells[start].get_left()
            right = cells[end].get_right()
            y = cells[start].get_bottom()[1] - 0.22
            line = Line([left[0], y, 0], [right[0], y, 0], stroke_color=MUTED, stroke_width=2)
            brackets.add(line)
        self.play(LaggedStart(*[FadeIn(b) for b in brackets], lag_ratio=0.15))
        self.wait(1.2)

        # 6. compile
        cap = self.swap_caption(cap, "Compile through the Tailwind v4 table")
        outputs = VGroup()
        for (start, end, cls), bracket in zip(SEGMENTS, brackets):
            cell = VGroup(*[TokenCell(part) for part in cls.split(" ")]).arrange(buff=0.05).scale(0.7)
            cell.next_to(bracket, DOWN, buff=0.25)
            outputs.add(cell)
        # stagger vertically so neighbours do not overlap
        for i, cell in enumerate(outputs):
            cell.shift(DOWN * (0.0 if i % 2 == 0 else 0.7))
        self.play(LaggedStart(*[ReplacementTransform(b.copy(), o) for b, o in zip(brackets, outputs)], lag_ratio=0.2), run_time=2.0)
        foot = note("every class validated against the compiled vocabulary")
        self.play(FadeIn(foot))
        self.wait(1.6)

        # 7. result
        cap = self.swap_caption(cap, "gpu-tailwind", FadeOut(foot), FadeOut(brackets))
        final = VGroup(*[TokenCell(part) for seg in SEGMENTS for part in seg[2].split(" ")]).arrange(buff=0.06).scale(0.78).move_to(DOWN * 1.0)
        self.play(ReplacementTransform(outputs, final), cells.animate.shift(UP * 0.4), run_time=1.4)
        foot = note(f"{SIZE} · runs in the browser on WebGPU")
        self.play(FadeIn(foot))
        self.wait(3)
