# Explainer videos

Every package ships a silent 45–60 s explainer in the style of
[gpu-lexer's](https://github.com/vercel-labs/gpu-lexer/tree/main/video): black frame,
Geist Mono, one accent color, the same token cells persisting from input to output.

Tooling is [Manim Community 0.19](https://docs.manim.community/) rendered headless
(Cairo, no GPU, no LaTeX, no display).

```bash
# once, on Amazon Linux / Fedora:
sudo dnf install -y cairo-devel pango-devel python3-devel pkgconf-pkg-config
# on Ubuntu: sudo apt-get install -y libcairo2-dev libpango1.0-dev

bash video/render.sh gpu-view draft   # 480p15, ~30 s render
bash video/render.sh gpu-view         # 1080p30, copies to apps/website/public/
```

## Files

- `style.py` — shared palette, `mono()`, `caption()`, `note()`, `TokenCell`, `activation_strip()`, `ExplainerScene`
- `scenes/<package>.md` — storyboard, written before the code (template in `scenes/TEMPLATE.md`)
- `<package>_pipeline.py` — one `ExplainerScene` subclass named `<Package>Pipeline`
- `assets/fonts/` — bundled Geist Mono
- `media/` — render output, gitignored

## Rules

- Storyboard first, in `scenes/<package>.md`, including the Accuracy Guardrails section.
- Only `Text` with the bundled font. Never `Code`, `Tex`, or `MathTex`.
- Six to eight beats of 5–12 s each, one top caption per beat, at most one muted footnote, end on a 2–3 s hold.
- Semantic colors appear only for the model's outputs. Everything else is white, muted grey, or the accent.
- Use `ReplacementTransform` so the same objects visibly carry through the pipeline.
- Real numbers on screen (parameter counts, sizes, latencies) must come from the package's model card.
