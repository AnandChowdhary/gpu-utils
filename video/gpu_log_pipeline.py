"""Explainer for gpu-log. Render: `bash video/render.sh gpu-log draft`."""

from manim import DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, LaggedStart, Rectangle, ReplacementTransform, Transform, VGroup

from style import ACCENT, MUTED, ExplainerScene, TokenCell, activation_strip, mono, note

LINE = "2024-01-15 10:30:00,123 [main] INFO com.example.Foo - Started in 12ms"
ROLE_COLORS = {"TS": "#22C55E", "LEVEL": "#F59E0B", "SOURCE": "#A855F7", "THREAD": "#06B6D4", "MSG": "#F43F5E"}
PARAMS = "154,332 parameters"
SIZE = "101.8 KB Brotli"
SPEED = "0.36 MB/s on CPU, GPU-batched"


def tokenize(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
        elif c.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
        elif c == " ":
            j = i
            while j < len(text) and text[j] == " ":
                j += 1
        else:
            j = i + 1
        out.append(text[i:j])
        i = j
    return out


def role_of(index: int, tokens: list[str]) -> str | None:
    pos = sum(len(t) for t in tokens[:index])
    if pos < 23:
        return "TS"
    if 25 <= pos < 29:
        return "THREAD"
    if 31 <= pos < 35:
        return "LEVEL"
    if 36 <= pos < 51:
        return "SOURCE"
    if pos >= 54:
        return "MSG"
    return None


class GpuLogPipeline(ExplainerScene):
    def construct(self):
        cap = self.swap_caption(None, "A log line")
        raw = mono(LINE, 20).move_to(UP * 0.4)
        self.play(FadeIn(raw, shift=DOWN * 0.08))
        self.wait(1.2)

        # Scene 2: tokens
        tokens = tokenize(LINE)
        cells = VGroup(*[TokenCell(t, muted=t.strip() == "") for t in tokens])
        cells.arrange(buff=0.05)
        cells.scale_to_fit_width(13.2).move_to(UP * 0.4)
        cap = self.swap_caption(cap, "Split into character-class runs", FadeOut(raw))
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.08) for c in cells], lag_ratio=0.02))
        foot = note("no vocabulary, no regex per format")
        self.play(FadeIn(foot))
        self.wait(1.0)

        # Scene 3: features
        cap = self.swap_caption(cap, "9 hashed feature ids per token", FadeOut(foot))
        strips = VGroup(*[activation_strip(c.source + str(i), c.get_center() + DOWN * 1.1, width=0.09, height=0.6) for i, c in enumerate(cells)])
        self.play(LaggedStart(*[FadeIn(s) for s in strips], lag_ratio=0.01))
        self.wait(1.0)

        # Scene 4: dilated windows
        cap = self.swap_caption(cap, "5 dilated conv blocks, dilation 1 → 16", FadeOut(strips))
        level_index = next(i for i, t in enumerate(tokens) if t == "INFO")
        center = cells[level_index]
        window = Rectangle(width=center.width + 0.1, height=0.75, stroke_color=ACCENT, stroke_width=2).move_to(center)
        self.play(FadeIn(window))
        for radius in (1, 3, 7, 15, 31):
            lo = max(0, level_index - radius)
            hi = min(len(cells) - 1, level_index + radius)
            span = VGroup(cells[lo], cells[hi])
            target = Rectangle(width=span.width + 0.1, height=0.75, stroke_color=ACCENT, stroke_width=2).move_to(span)
            self.play(Transform(window, target), run_time=0.55)
        foot = note(f"{PARAMS}, int6 quantised")
        self.play(FadeIn(foot))
        self.wait(0.8)

        # Scene 5: tags
        cap = self.swap_caption(cap, "One role per token", FadeOut(window), FadeOut(foot))
        anims = []
        for i, c in enumerate(cells):
            role = role_of(i, tokens)
            if role and c.source.strip():
                target = c.copy()
                target.box.set_stroke(ROLE_COLORS[role])
                target.glyph.set_color(ROLE_COLORS[role])
                anims.append(Transform(c, target))
        self.play(LaggedStart(*anims, lag_ratio=0.01))
        legend = VGroup(*[mono(r, 15, col) for r, col in ROLE_COLORS.items()]).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.7)
        kind = mono("kind: entry", 15, MUTED).move_to(DOWN * 1.3)
        self.play(FadeIn(legend), FadeIn(kind))
        self.wait(1.0)

        # Scene 6: compiler
        cap = self.swap_caption(cap, "Compiled into a typed record", FadeOut(legend), FadeOut(kind))
        rows = [
            ("timestamp", "2024-01-15T10:30:00.123", "TS"),
            ("level", "info", "LEVEL"),
            ("source", "com.example.Foo", "SOURCE"),
            ("thread", "main", "THREAD"),
            ("message", "Started in 12ms", "MSG"),
        ]
        record = VGroup()
        for key, value, role in rows:
            line = VGroup(mono(f"{key}:", 19, MUTED), mono(value, 19, ROLE_COLORS[role])).arrange(RIGHT, buff=0.25)
            record.add(line)
        record.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(UP * 0.2)
        groups = {}
        for i, c in enumerate(cells):
            role = role_of(i, tokens)
            groups.setdefault(role, VGroup()).add(c)
        anims = []
        for (key, value, role), line in zip(rows, record):
            anims.append(ReplacementTransform(groups[role], line))
        anims.append(FadeOut(groups[None]))
        self.play(*anims, run_time=1.4)
        self.wait(1.0)

        # Scene 7: batching
        cap = self.swap_caption(cap, "Thousands of lines per GPU dispatch")
        small = record.copy().scale(0.55).to_edge(RIGHT, buff=0.6)
        wall = VGroup(*[mono("▮" * (18 + (i * 7) % 11), 13, MUTED) for i in range(12)]).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_edge(LEFT, buff=0.8)
        ticks = VGroup(*[mono(str(i), 11, ACCENT).next_to(w, LEFT, buff=0.15) for i, w in enumerate(wall)])
        self.play(Transform(record, small), FadeIn(wall), FadeIn(ticks))
        foot = note("JSON and logfmt lines are parsed deterministically and skip the model")
        self.play(FadeIn(foot))
        self.wait(1.2)

        # Scene 8: hold
        cap = self.swap_caption(cap, "gpu-log", FadeOut(wall), FadeOut(ticks), FadeOut(foot))
        end = note(f"{PARAMS} · {SIZE} · {SPEED}")
        self.play(FadeIn(end))
        self.wait(2.0)
