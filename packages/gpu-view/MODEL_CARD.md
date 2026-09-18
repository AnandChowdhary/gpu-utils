# Model card: gpu-view

## Task
Natural language to table view specs: filter, sort, group, aggregate, limit, chart. Input is
a search-bar phrase plus the caller's schema; output is a typed spec with character spans and
diagnostics. The model is schema-blind: it tags tokens with roles and clause boundaries using
anonymous schema-membership features, and a deterministic compiler resolves everything else.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime`), whitespace dropped before the model
- Features per token (__ROWS__ rows, up to 28 active): shape, length bucket, word hash (128),
  consonant-skeleton hash (64), closed task lexicon (__KEYWORDS__ words + none), 12 flags
  (digits, punctuation, casing, position, year-like, numeric suffix, comparative/superlative
  endings), and the schema-membership rows: field kind / begin-inside / match quality (exact,
  stem, inflection, prefix, typo) / alias, enum match / begin-inside / unique owner / owned by
  nearest preceding or following field, kind of and distance to the nearest field match before
  and after, relative position
- Model: shared scan family `ScanTagger(feature_rows, 32, 15)` (`gpu_utils_training.models`):
  32-dim summed embeddings → one bidirectional gated affine scan layer
  `h[t] = a[t]·h[t−1] + (1−a[t])·tanh(u[t])` (masked Hillis-Steele prefix scan in training,
  per-channel walk in the canonical WGSL `scan` pass) → residual 5-tap depthwise convolution →
  mean-pooled context → head (64 relu) → 15 logits: 14 roles + 1 clause-boundary logit;
  argmax decode per column group
- Roles: `O FIELD OP VALUE TIME_VALUE CONJ NEG SORT_FIELD SORT_DIR GROUP_FIELD AGG_FN AGG_FIELD LIMIT CHART`
- Parameters: __PARAMS__
- Quantization: int6 symmetric per-tensor, quantization-aware training from epoch 1 (straight-through)
- v1 (0.1) used a package-specific tagger (5-tap conv before the scan, gated global context,
  33,087 params); v2 moves to the shared family with no package kernel

## Training data
Entirely synthetic; no external datasets were used (there is no permissively licensed corpus
of search phrases paired with view specs and schemas, and the schema-blind setup needs the
schema that produced each phrase). `training/gpu_view/generate.py` renders 160,000 phrases
(seed 0) from random schemas sampled out of eight domain pools (e-commerce, CRM, issues,
music, HR, logistics, analytics, finance; 4–9 fields each, aliases randomly hidden, enum
values subsampled), with hundreds of clause templates, synonyms for every operator /
aggregate / sort / group / chart / limit word, plural / prefix / one-edit corruptions of field
words (only when they still resolve), typos in function words, quoted values, casing noise and
punctuation noise. Gold roles come from the renderer's structure; the TypeScript compiler
round-trips the gold roles to the gold spec on 100% of the evaluation sets
(`test/roundtrip.test.ts`), so model metrics below measure the model, not the compiler.

Four whole domains (recipes, real estate, education, travel) are held out; their field names,
aliases and enum values share no content word with the training pools (`schema.assert_disjoint`).

## Evaluation
Spec exact match compares filters, sort, groupBy, aggregate, limit, chart and granularity;
spans included for the generated sets, ignored for the hand-written sets (specs were written
without offsets). Numbers are from the promoted checkpoint on the CPU path (`pnpm test`).

| Set | Size | Metric | v1 (0.1) | v2 |
|---|---|---|---|---|
| in-domain (training domains, fresh seed) | 200 | spec exact match, with spans | 96.5% | __INDOMAIN__ |
| held-out (four unseen domains, disjoint vocabulary) | 400 | spec exact match, with spans | 91.0% | __TRANSFER__ |
| held-out | 2,000 | token role accuracy / boundary accuracy | 99.27% / 99.45% | __TOKACC__ / __BNDACC__ |
| unfamiliar v1 (65 hand-written phrases, 4 schemas the generator never saw) | 65 | spec exact match | 55.4% | __UNFAMILIAR1__ |
| unfamiliar v2 (67 hand-written phrases, 4 further schemas, written before v2 was evaluated) | 67 | spec exact match | — | __UNFAMILIAR2__ |

The v1 and v2 generated sets differ (the v2 generator covers more categories), so the
generated-set columns are not a like-for-like comparison; the unfamiliar sets are.
`eval/unfamiliar-v1.json` (podcasts, wine cellar, repositories, greenhouse) was analysed
after v1 and drove the v2 coverage categories, so it is contaminated as a held-out set;
`eval/unfamiliar-v2.json` (conference talks, workouts, art auction, restaurant
reservations) was written after the v2 generator changes and before the v2 model was
evaluated, and was not used for tuning.

v2 coverage categories (each addressed in the generator, lexicon or compiler, not per phrase):
comparative and superlative adjectives resolved through field aliases with a polarity lexicon
(`cheapest first`, `taller than 50 cm`, `highest rated`); unit suffixes on numbers (`50 cm`,
`2 kg`, `usd 50`); month-day-year dates and seasons (`september 10 2026`, `10th of sep`,
`this spring`); year-like numeric fields (`vintage between 2015 and 2020`); `top N` stranded
before its sort field; `count of X by Y` grouping; enum values overriding a carrier noun that
matches a text field (`business or tech episodes`); entity nouns colliding with field aliases;
`primary: true` date fields.

Failure analysis of the v2 unfamiliar set:

__UNFAMILIAR_ANALYSIS__

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 29.6 KiB (budget 39.1 KiB) |
| Weights (int6 text) | 33,087 chars |
| Import + weight decode (Node 24, Xeon 2.9 GHz) | ~5 ms |
| First parse (JIT warm-up) | ~15 ms |
| Warm CPU parse, 19-token phrase | ~0.85 ms |
| CPU batch, 1000 phrases | ~640 ms |
| WebGPU | kernel verified against the CPU path (max |Δ| 7.6e-6 over 26 phrases, one dispatch) on Mesa llvmpipe via wgpu-py; not timed on real hardware |

## Limitations and intended use
Intended for search bars and view builders where the app validates the spec before running
it. Outputs are probabilistic; treat the spec as a proposal and show diagnostics to the user.
Known gaps: superlatives and comparatives outside the lexicon (`cheapest`, `tallest`),
inflections the matcher cannot bridge (`unopened`, `rated`), units after numbers (`50 cm`),
dates outside the resolver (`september 10 2026`, `this spring`), clause-level `or`, negated
`contains` / `between` / date ranges, `top N` inferring a sort from an aggregate, phrases
whose time filter follows a group unit are occasionally mis-segmented, English only.

## Checkpoint
- Promoted: `runs/default` — 2026-09-18, seed 0, 160,000 samples, 14 epochs, batch 128,
  AdamW lr 3e-3 one-cycle, QAT from epoch 2, 2 CPU threads, 694 s wall clock
- Training command: `pnpm train` (`uv run python -m gpu_view.train`), export with `pnpm export`
