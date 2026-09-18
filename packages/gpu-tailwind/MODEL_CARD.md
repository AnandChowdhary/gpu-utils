# Model card: gpu-tailwind

## Task
Natural language to Tailwind CSS v4 utility classes. Input: a phrase such as
"card with rounded corners, subtle shadow, blue on hover, hidden on mobile". Output:
`{ classes, groups: [{ span, text, classes, variant? }], diagnostics }` where every class
is validated against a vocabulary compiled from the Tailwind v4 default theme.

## Design choice: tagging + compiler (chosen) vs. multi-label head

The brief allowed two designs. Both were built and evaluated on the same data (numbers
below are on the v2 generator's held-out set):

| Design | Params | Held-out exact-set match | Notes |
|---|---|---|---|
| Role tagger + deterministic compiler (shipped) | 45,394 | 89.7% | composes colour × shade × alpha × property, arbitrary values (`p-[13px]`), gradients, stacked variants; all semantics in code |
| Mean-pooled multi-label head over the class vocabulary (`baseline_multilabel.py`) | 189,536 | 4.5% | vocabulary capped at classes seen ≥20× (1,712 classes); 47.0% of held-out classes are unreachable by construction; cannot bind a variant to one segment |

The multi-label head is structurally unable to express the long tail (any number × any
unit, 286 colour tokens × 12 colour properties × 11 alphas × 37 variants) and its head
alone would exceed the whole tagger's parameter count, so the tagging design was chosen;
it also keeps every semantic decision inspectable in `src/compile.ts`.

## Architecture

The shared scan family, `gpu_utils_training.models.ScanTagger(ROWS, 24, 10)`. The package
ships no model code and no shader of its own: `src/cpu.ts` calls `scanTaggerForward` from
`@gpu-utils/runtime` and `src/gpu.ts` runs the canonical
`packages/runtime/src/wgsl/scan_tagger.wgsl` kernel through `runScanTagger`.

- Tokenizer: character-class runs (shared `@gpu-utils/runtime`); 7 sparse hashed features
  per token (word 1024, consonant skeleton 256, prefix 128, suffix 128, shape 8, length 16,
  class 5) into one 1,565 × 24 embedding table; no learned vocabulary.
- Embedding: 24 dims (sum of the 7 rows).
- Sequence mixing: one bidirectional gated affine scan, `a = σ(Wa e + ba)`,
  `u = tanh(Wu e + bu)`, `h_t = a ⊙ h_{t-1} + (1 − a) ⊙ u` forward and backward, then a
  residual 5-tap depthwise conv and a mean-pooled context vector.
- Head: two-layer, 32 hidden → 10 logits: 9 BIO role labels (`O`, `B/I-PROP`, `B/I-VAL`,
  `B/I-VAR`, `SEP`, `NEG`) decoded with a constrained Viterbi, plus one segment-boundary
  logit. The extra tag column is the family's sanctioned way to add a per-token head; the
  decoder splits it off (`src/decode.ts`).
- Parameters: 45,394.
- Quantization: int6 symmetric per-tensor, quantization-aware training (fake-quant with a
  straight-through estimator from epoch 1 onward). Fixtures are computed from the
  dequantized weights, so TypeScript (CPU) and WGSL match PyTorch at 1e-4; the measured
  WGSL-vs-CPU worst case on Mesa lavapipe is 1.5e-5.
- Compiler: `src/compile.ts` + `src/decode.ts`, mirrored by
  `training/gpu_tailwind/{semantics,pairing,lexicon}.py`.

## Training data
- No real dataset was used; no permissively licensed corpus of NL→Tailwind pairs exists
  that we could find, and the held-out generated set is therefore not a substitute for
  real usage data (see "unfamiliar" below).
- Tailwind v4 default theme (`tailwindcss@4.3.3/theme.css`, MIT, Tailwind Labs) vendored
  as `training/data/tailwind-theme.txt` and compiled by `build_table.py` into
  `src/table.json` together with a hand-written lexicon of ~510 property phrasings,
  ~1,700 value phrasings, ~1,400 variant phrasings, presets, separators and negations.
- Synthetic generator (`data.py`, seed 1): 130,000 training phrases of 1–5 segments; each
  segment has 1–3 units (property+value, standalone value, negated property, preset noun,
  a literal class, a two-colour "X on a Y background" unit or a gradient), an optional
  variant phrase, glue words, intros/outros, commas / "and" / "with" / ";" / "+"
  separators, random casing, British spellings, single-edit typos on content words, number
  words, shade and alpha spellings ("a lighter shade of blue", "black at 30 percent"),
  attached/detached units and occasional double spaces. Gold classes are produced by the
  Python mirror of the compiler on the clean pieces, so generator and runtime agree by
  construction.
- Held-out: 6,000 phrases from the same generator with seed 2, generated **after** the
  training split and excluding every training phrase and every hand-written evaluation
  phrase (before v2 this was not enforced and 172 held-out phrases also occurred in
  training).
- Generator/compiler parity: the committed 400-phrase oracle sample (`test/oracle.test.ts`)
  is 400/400. Over the full 6,000-phrase held-out set the TypeScript compiler reproduces
  the Python gold on 5,970 (99.5%); the 30 residual disagreements are all cases where the
  generator's injected typo has more than one vocabulary word at edit distance 1 and the
  two spell-checkers pick differently. That 0.5% is the ceiling on held-out exact match.

## Evaluation

Three sets, all scored with the full pipeline (`pnpm eval`, `training/runs/eval.json`):

- **Held-out (generated, seed 2, 3,000 phrases).** Same distribution as training.
- **Unfamiliar v1 (66 hand-written phrases, `eval/unfamiliar-v1.json`).** Frozen from the
  v1 round. **Treat as contaminated**: its misses are what the v2 generator was widened to
  cover (gradients, alpha, shade wording, "X on a Y background", leading variants). It is
  reported for continuity, not as evidence of generalization.
- **Unfamiliar v2 (70 hand-written phrases, `eval/unfamiliar-v2.json`).** Written by hand
  before any v2 model was trained and never used for tuning. This is the number to quote.

Neither unfamiliar set appears in the training or held-out splits (checked by exact text
match in `data.py`).

### v1 vs v2, same evaluation sets, same scorer

"v1" is the package as merged in #8 (custom `nn.Module` + hand-written shader, 45,218
params); "v2" is this branch. Both were run through `test/eval.test.ts` on the sets below.

| Set | Metric | v1 | v2 |
|---|---|---|---|
| Held-out (v2 generator, n=3,000) | exact-set match | 53.2% | **89.7%** |
| | class F1 | 71.6% | **96.7%** |
| Unfamiliar v1 (n=66, contaminated) | exact-set match | 37.9% | **53.0%** |
| | class F1 | 74.2% | **77.0%** |
| Unfamiliar v2 (n=70, clean) | exact-set match | 27.1% | **52.9%** |
| | class F1 | 60.1% | **80.6%** |

The held-out row measures both models on the *v2* generator, which is much wider than v1's
(gradients, alpha, two-colour units, number words, heavier casing/typo noise), so v1 scores
far below the 90.3% it reached on its own narrower held-out set. The two unfamiliar sets
are fixed files and are directly comparable.

Macro (per distinct class) averages for v2: held-out P 94.2% / R 94.0%,
unfamiliar v1 P 65.3% / R 64.3%, unfamiliar v2 P 67.3% / R 68.1%.

### Tag level (held-out, 6,000 phrases, `uv run python -m gpu_tailwind.evaluate`)

| Metric | v1 | v2 |
|---|---|---|
| Token accuracy | 98.8% | 99.04% |
| Whole-sequence tag accuracy | 82.4% | 85.3% |
| Boundary accuracy | 99.96% | 99.94% |
| PROP span F1 | 97.3% | 97.6% |
| VAL span F1 | 95.2% | 96.3% |
| VAR span F1 | 95.9% | 97.4% |

(v1's tag numbers are on the v1 generator's held-out set, v2's on the harder v2 set.)

### What still fails

All 33 misses on unfamiliar v2 are in `training/runs/misses-unfamiliarV2.txt`, written by
`pnpm eval`; nothing below was fixed or tuned after seeing them.

- **A bare colour still defaults to a background.** 7 misses. "font light, italic, muted
  gray 400" → `bg-gray-400`; "bold 3xl heading, tight tracking, gray 900" → `bg-gray-900`;
  "the placeholder text should be gray 400" → `placeholder:bg-gray-400`. A bare colour only
  becomes `text-*` when its segment also holds a text property to pair it with, or when the
  segment holds exactly two bare colours ("black on yellow" → `text-black bg-yellow-500`);
  a trailing colour usually ends up in a segment of its own and falls back to `bg-*`.
- **Only the first two breakpoint clauses survive.** 4 misses. "two column grid on tablets,
  four on desktop, single column on phones" stops after `md:`; "text small on mobile, base
  on tablet, large on desktop" drops `lg:`. Third and later variant segments lose their
  `VAR` span.
- **Variant identity and direction.** 5 misses. "when its parent card is hovered" →
  `hover:` rather than `group-hover:`; "lift the shadow to extra large when hovered" →
  `xl:hover:shadow-md` (the size word is read as a breakpoint); "only show it on desktop" →
  `lg:block` rather than `max-lg:hidden`.
- **Spurious near-duplicates from over-tagging.** 5 misses. `overflow-y-auto overflow-auto`,
  `whitespace-normal whitespace-pre-wrap`, `transition transition-all`, and a stray
  `border-gray-200` default alongside `border-t-teal-200`. Precision on unfamiliar v2 is
  82.2% while exact match is 52.9%: most outputs are right plus one extra or one missing
  class.
- **Idioms with no lexicon entry.** 6 misses. "truncate after one line with an ellipsis",
  "text center on mobile only", "vertically centered, horizontally spaced between",
  "cursor grab, select none", "12 by 12" as a size.
- **Alpha vs. opacity.** "white background at 80% opacity" → `bg-white opacity-80` instead
  of `bg-white/80`; the compiler only folds the alpha in when it is attached to the colour
  phrase.
- **Preset over-trigger.** "the button reveals a shadow when its parent card is hovered"
  fires the whole `card` preset because the word *card* is present.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights and table) | 54.9 KiB (56,254 B) of a 58.6 KiB (60,000 B) budget |
| Weights (int6 text) | 45,394 B raw, 25.3 KiB Brotli |
| Class table (`src/table.json`) | 77.7 KiB raw, 13.4 KiB Brotli |
| Cold start (device + pipelines + upload), estimate | ~40-80 ms |
| Warm WebGPU call, 300 tokens, estimate | ~1-2 ms compute + ~1-3 ms readback |
| CPU path, 15-token phrase (Node 24, shared box) | 0.48 ms (v1 measured 0.39 ms in the same session) |

WebGPU numbers are estimates from the runtime's readback-bound behaviour on this class of
model (no GPU browser on the training box); the WGSL kernel is verified against the
fixtures on a lavapipe adapter (`training/tests/test_wgsl.py`).

## Limitations and intended use
- Opinionated defaults: `rounded` → `rounded-lg`, `shadow` → `shadow-md`, `border` →
  `border`, `padding` → `p-4`, a bare colour → `bg-*` (or `text-*` when a text property is
  in the same segment), `on small screens` / `on mobile` → `max-sm:`, `on desktop` → `lg:`,
  `only on mobile` → `sm:hidden`, `ring` → `ring-2`, `centered` → `flex items-center
  justify-center`. The compiler is the place to change them.
- Values are paired with the nearest compatible property in the same segment; phrases that
  rely on long-distance agreement ("make the border and the text both blue") only colour
  the nearer one.
- A leading variant phrase ("on hover, blue and underlined") scopes over the following
  segments until the next variant phrase. Three or more breakpoint clauses in one sentence
  are unreliable (see above).
- Relative colours ("darker on hover") need the base colour in the same phrase; without it
  the value is reported as a diagnostic.
- Not covered: arbitrary selectors/variants, custom theme tokens, `!important`, negative
  values beyond `-m*`, filters other than blur/grayscale, transforms other than
  scale/rotate, container queries, `backdrop-*`.
- Outputs are probabilistic tag predictions; the compiler guarantees validity, not intent.
  Review the classes before shipping them.

## Checkpoint
- Promoted: seed 0, epoch 29 of 30, step 30,480, 2026-09-18 (16.8 min on 2 CPU threads)
- Training command: `pnpm train` (`uv run python -m gpu_tailwind.train --minutes 24 --epochs 30`,
  seed 0, batch 128, AdamW 4e-3 cosine, QAT from epoch 1, 2 CPU threads).
