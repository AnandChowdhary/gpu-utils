# Model card: gpu-tailwind

## Task
Natural language to Tailwind CSS v4 utility classes. Input: a phrase such as
"card with rounded corners, subtle shadow, blue on hover, hidden on mobile". Output:
`{ classes, groups: [{ span, text, classes, variant? }], diagnostics }` where every class
is validated against a vocabulary compiled from the Tailwind v4 default theme.

## Design choice: tagging + compiler (chosen) vs. multi-label head

The brief allowed two designs. Both were built and evaluated on the same data:

| Design | Params | Held-out exact-set match | Notes |
|---|---|---|---|
| Role tagger + deterministic compiler (shipped) | 45,218 | 90.3% | composes colour × shade × property, arbitrary values (`p-[13px]`), stacked variants; all semantics in code |
| Mean-pooled multi-label head over the class vocabulary (`baseline_multilabel.py`) | 174,391 | 8.6% | vocabulary capped at classes seen ≥20× (1,479 classes); 34.1% of held-out classes are unreachable by construction; cannot bind a variant to one segment |

The multi-label head is structurally unable to express the long tail (any number × any
unit, 286 colour tokens × 12 colour properties × 37 variants) and its head alone would
exceed the whole tagger's parameter count, so the tagging design was chosen; it also keeps
every semantic decision inspectable in `src/compile.ts`.

## Architecture
- Tokenizer: character-class runs (shared `@gpu-utils/runtime`); 7 sparse hashed features
  per token (word 1024, consonant skeleton 256, prefix 128, suffix 128, shape 8, length 16,
  class 5) into one 1,565 × 24 embedding table; no learned vocabulary.
- Embedding: 24 dims (sum of the 7 rows).
- Sequence mixing: two gated affine scans (forward and backward), `a = σ(Wa e + ba)`,
  `u = tanh(Wu e + bu)`, `h_t = a ⊙ h_{t-1} + (1 − a) ⊙ u`; `x = [e; h_f; h_b]` (72),
  depthwise conv (kernel 3) plus a mean-pooled global context vector projected into the head.
- Head: ReLU(72 → 32) → 10 logits: 9 BIO role labels (`O`, `B/I-PROP`, `B/I-VAL`,
  `B/I-VAR`, `SEP`, `NEG`) decoded with a constrained Viterbi, plus one segment-boundary logit.
- Parameters: 45,218.
- Quantization: int6 symmetric per-tensor, quantization-aware training (fake-quant with a
  straight-through estimator from epoch 1 onward). Fixtures are computed from the
  dequantized weights, so TypeScript (CPU) and WGSL match PyTorch at 1e-4.
- Compiler: `src/compile.ts` + `src/decode.ts`, mirrored by
  `training/gpu_tailwind/{semantics,pairing,lexicon}.py`.

## Training data
- No real dataset was used; no permissively licensed corpus of NL→Tailwind pairs exists
  that we could find, and the held-out generated set is therefore not a substitute for
  real usage data (see "unfamiliar" below).
- Tailwind v4 default theme (`tailwindcss@4.3.3/theme.css`, MIT, Tailwind Labs) vendored
  as `training/data/tailwind-theme.txt` and compiled by `build_table.py` into
  `src/table.json` together with a hand-written lexicon of ~480 property phrasings,
  ~1,650 value phrasings, ~1,400 variant phrasings, presets, separators and negations.
- Synthetic generator (`data.py`, seed 1): 130,000 training phrases of 1–5 segments; each
  segment has 1–3 units (property+value, standalone value, negated property, preset noun or
  a literal class), an optional variant phrase (32%), glue words, intros/outros, commas /
  "and" / "with" / ";" / "+" separators, random casing, British spellings (20% of eligible
  words), single-edit typos on 5% of content words, number words, attached/detached units
  and occasional double spaces. Gold classes are produced by the Python mirror of the
  compiler on the clean pieces, so generator and runtime agree by construction
  (`test/oracle.test.ts`, 400/400).
- Held-out: 6,000 phrases from the same generator with seed 2 (3,000 used for metrics).
- Unfamiliar: 66 hand-written phrases (`training/data/unfamiliar.json`) with sentence
  shapes and vocabulary the generator does not produce ("three column grid on desktop, one
  column on phones", "when the mouse is over it, lift the shadow to large", "zebra striped
  rows with a slate 50 background"). Written before training and never used for tuning.

## Evaluation

Tag level (held-out, 6,000 phrases, `uv run python -m gpu_tailwind.evaluate`):

| Metric | Score |
|---|---|
| Token accuracy | 98.8% |
| Whole-sequence tag accuracy | 82.8% |
| Boundary accuracy | 99.96% |
| PROP span F1 | 97.3% |
| VAL span F1 | 95.2% |
| VAR span F1 | 95.9% |

Class level (full pipeline, `pnpm eval`, `training/runs/eval.json`):

| Set | Size | Exact-set match | Class precision | Class recall | Class F1 |
|---|---|---|---|---|---|
| Held-out (generated, seed 2) | 3,000 | 90.3% | 97.6% | 96.6% | 97.1% |
| Unfamiliar (hand-written) | 66 | 37.9% | 77.6% | 71.1% | 74.2% |

Per-class precision/recall are micro-averaged over every emitted/expected class string;
macro averages over distinct classes: held-out P 94.9% / R 94.7%,
unfamiliar P 60.4% / R 60.4%.

What the unfamiliar set shows (all 41 misses are in `training/runs/unfamiliar_misses.txt`;
nothing below was fixed or tuned after seeing them):

- **Variant scope errors dominate.** On phrasings the generator never produced, the tagger
  attaches a variant to the wrong segment or invents one from a lone keyword: "overlay
  covering the parent" becomes `group-hover:` because of *parent*; "heading: 4xl" becomes
  `first:` because of *heading*; "when the mouse is over it, lift the shadow" loses `hover:`
  entirely; "zebra striped rows with a slate 50 background" drops `even:`. 13 of 41 misses.
- **Value/property confusions on rare pairings.** "white text on a gray 900 background"
  → `dark:bg-white` (the colour binds to the nearer *background*); "text emerald 600" →
  `bg-emerald-600` (the standalone-colour default is a background); "scale up 5%" →
  `scale-5`. The compiler's nearest-property rule is right for the generator's
  distribution and wrong for some natural English word orders. 11 misses.
- **Missing vocabulary.** "hairline", "translucent ... backdrop" (`bg-black/50` is not a
  form the compiler emits), "spin animation" phrasing, "lift the shadow", "hide it when
  printing" ("hide it" is not a value phrase). 9 misses.
- **Spurious extra classes** from over-eager tagging of glue words ("so it sits above" →
  `transition-all`, "all round" → `rounded-full`, "gap of 6" also emitting `grid-cols-2`).
  8 misses.

Per-class precision (77.6%) is well above exact-set match (37.9%): most outputs contain
the right classes plus one wrong or missing one. The held-out generated set is therefore
an optimistic number; the unfamiliar set is the one to quote. The obvious next step is a
generator pass over exactly these sentence shapes (variant phrase first + imperative verb,
"X on a Y background", noun-phrase descriptions), which we deliberately did not do before
reporting.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights and table) | 52.3 KiB |
| Weights (int6 text) | 44.2 KiB raw |
| Cold start (device + pipelines + upload), estimate | ~40-80 ms |
| Warm WebGPU call, 300 tokens, estimate | ~1-2 ms compute + ~1-3 ms readback |
| CPU path, 15-token phrase (Node 24) | 0.38 ms |

WebGPU numbers are estimates from the runtime's readback-bound behaviour on this class of
model (no GPU browser on the training box); the WGSL kernels are verified against the
fixtures on a lavapipe adapter (`training/tests/test_wgsl.py`).

## Limitations and intended use
- Opinionated defaults: `rounded` → `rounded-lg`, `shadow` → `shadow-md`, `border` →
  `border`, `padding` → `p-4`, a bare colour → `bg-*` (or `text-*` when a text property is
  in the segment), `on small screens` / `on mobile` → `max-sm:`, `on desktop` → `lg:`,
  `only on mobile` → `sm:hidden`, `ring` → `ring-2`, `centered` → `flex items-center
  justify-center`. The compiler is the place to change them.
- Values are paired with the nearest compatible property in the same segment; phrases that
  rely on long-distance agreement ("make the border and the text both blue") only colour
  the nearer one.
- Relative colours ("darker on hover") need the base colour in the same phrase; without it
  the value is reported as a diagnostic.
- Not covered: gradients, arbitrary selectors/variants, custom theme tokens, `!important`,
  negative values beyond `-m*`, filters other than blur/grayscale, transforms other than
  scale/rotate, container queries.
- Outputs are probabilistic tag predictions; the compiler guarantees validity, not intent.
  Review the classes before shipping them.

## Checkpoint
- Promoted: seed 0, epoch 28 of 30, step 29,464, 2026-09-18 (13.4 min on 2 CPU threads)
- Training command: `pnpm train` (`uv run python -m gpu_tailwind.train --minutes 16 --epochs 30`,
  seed 0, batch 128, AdamW 4e-3 cosine, QAT from epoch 1, 2 CPU threads).
