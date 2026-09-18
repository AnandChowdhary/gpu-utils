# Model card: gpu-view

## Task
Natural language to table view specs: filter, sort, group, aggregate, limit, chart. Input is
a search-bar phrase plus the caller's schema; output is a typed spec with character spans and
diagnostics. The model is schema-blind: it tags tokens with roles and clause boundaries using
anonymous schema-membership features, and a deterministic compiler resolves everything else.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime`), whitespace dropped before the model
- Features per token (748 rows, up to 29 active): shape, length bucket, word hash (128),
  consonant-skeleton hash (64), closed task lexicon (469 words + none), 12 flags
  (digits, punctuation, casing, position, year-like, numeric suffix, comparative/superlative
  endings), and the schema-membership rows: field kind / begin-inside / match quality (exact,
  stem, inflection, prefix, typo) / alias / reached through a negation prefix, enum match /
  begin-inside / unique owner / owned by nearest preceding or following field, kind of and
  distance to the nearest field match before and after, relative position
- Model: shared scan family `ScanTagger(feature_rows, 32, 15)` (`gpu_utils_training.models`):
  32-dim summed embeddings → one bidirectional gated affine scan layer
  `h[t] = a[t]·h[t−1] + (1−a[t])·tanh(u[t])` (masked Hillis-Steele prefix scan in training,
  per-channel walk in the canonical WGSL `scan` pass) → residual 5-tap depthwise convolution →
  mean-pooled context → head (64 relu) → 15 logits: 14 roles + 1 clause-boundary logit;
  argmax decode per column group
- Roles: `O FIELD OP VALUE TIME_VALUE CONJ NEG SORT_FIELD SORT_DIR GROUP_FIELD AGG_FN AGG_FIELD LIMIT CHART`
- Parameters: 37,775 (23,936 embedding + 13,839 scan/conv/head)
- Quantization: int6 symmetric per-tensor, quantization-aware training from epoch 1 (straight-through)
- v1 (0.1) used a package-specific tagger and its own `shader.wgsl` (5-tap conv before the
  scan, gated global context, 33,087 params); v2 moves to the shared scan family and the
  canonical runtime kernel, with no package kernel

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
without offsets). Every number below is the *shipped package* — model plus compiler — run on
the CPU path (`pnpm test`). The v1 column is the released 0.1 package (its own model, its own
compiler) re-run on exactly these sets, so the two columns are like-for-like.

| Set | Size | Metric | v1 (0.1) | v2 |
|---|---|---|---|---|
| in-domain (training domains, fresh seed) | 200 | spec exact match, with spans | 82.5% | 97.5% |
| held-out (four unseen domains, disjoint vocabulary) | 400 | spec exact match, with spans | 73.3% | 88.3% |
| held-out | 2,000 | token role accuracy / boundary accuracy | — | 99.30% / 99.43% |
| unfamiliar v1 (65 hand-written phrases, 4 schemas the generator never saw) | 65 | spec exact match | 55.4% | 73.8% |
| unfamiliar v2 (67 hand-written phrases, 4 further schemas) | 67 | spec exact match | 46.3% | 67.2% |
| unfamiliar v3 (67 hand-written phrases, 4 further schemas) | 67 | spec exact match | 50.7% | 71.6% |

The generated sets are produced by the v2 generator, which covers more categories than the
v1 one; v1 scored 96.5% / 91.0% on the narrower sets it was released with, and 82.5% / 73.3%
on these. Role-level accuracy is not comparable across versions (different label stream).

Provenance of the three hand-written sets, which is what makes them worth reading:

- `eval/unfamiliar-v1.json` (podcasts, wine cellar, repositories, greenhouse) was analysed
  after v1 shipped and drove every v2 coverage category. **It is contaminated** as a held-out
  set and is reported only for continuity with the v1 model card.
- `eval/unfamiliar-v2.json` (conference talks, workouts, art auction, restaurant
  reservations) was written after the v2 generator changes and before the v2 model was
  trained, but eight of its failures were read during v2 development, so it is
  *partially exposed*: no fix targeted them, and none of the categories below came from it.
- `eval/unfamiliar-v3.json` (library loans, veterinary visits, bird sightings, board game
  collection) was written and committed before the model was ever run on those schemas, and
  the error analysis that drove this round used unfamiliar-v1 only. It is the **clean**
  number: 50.7% -> 71.6%. The failures listed below were read only after that number was
  final, and nothing was changed afterwards.

v2 coverage categories (each addressed in the generator, lexicon or compiler, not per phrase):
comparative and superlative adjectives resolved through field aliases with a polarity lexicon
(`cheapest first`, `taller than 50 cm`, `highest rated`); a negation prefix glued onto a
boolean field word (`unarchived`, `nonbillable`, `inactive`), matched by stripping
un/non/dis/im/ir/in when the base resolves to a boolean field and surfaced to the model as
its own feature row; a unit noun that repeats the field after a value (`longer than 60
minutes`); unit suffixes on numbers (`50 cm`, `2 kg`, `usd 50`); month-day-year dates and
seasons (`september 10 2026`, `10th of sep`, `this spring`); year-like numeric fields with
polarity bounds (`vintage between 2015 and 2020`, `2019 or older`); `top N` stranded before
its sort field; `count of X by Y` grouping; enum values overriding a carrier noun that matches
a text field (`business or tech episodes`); entity nouns colliding with field aliases;
`primary: true` date fields; negated relative windows (`not updated in the last 30 days`).

Compiler-side, a normalisation pass reconciles clauses that the boundary head split through
the middle of one constraint: same-field `eq` filters on an enum or text field merge into one
value list, a `not_empty` implied by another filter on the same field is dropped, a field
sorted twice keeps the direction of the later clause, and duplicate aggregates collapse. That
pass alone moved unfamiliar-v1 from 67.7% to 73.8% and unfamiliar-v3 from 64.2% to 65.7%;
retraining on the widened generator took v3 the rest of the way to 71.6%.

Failure analysis of the clean unfamiliar-v3 set (19 of 67 phrases):

| Class | Cases | Example |
|---|---|---|
| Comparative / superlative that is not a field word | 5 | `longest books first`, `cats heavier than 5 kg`, `most expensive visits first` |
| Free-text value the model does not tag as a value | 4 | `titles containing catan`, `birds spotted by Hannah in devon` |
| Number whose only field cue is a non-field unit noun | 3 | `party games under 30 minutes`, `coop games with at most 4 players` |
| Bare time phrase dropped after an enum clause | 3 | `emergency visits this week`, `sightings in wetland habitats last spring` |
| Verb-phrase negation | 1 | `strategy games i do not own` |
| Two booleans joined by `and` (second read as a group) | 1 | `rare and verified sightings, newest first` |
| `before`/`after` on a year-like numeric field compiles to `eq` | 1 | `strategy or abstract games published before 2010` |
| Gold/compiler disagreement on `from <month>` (`gte` vs window) | 1 | `unverified sightings from this month` |

The first three classes are the schema-blind design showing its edge: with no `long` alias on
`pages` and no field called `minutes`, nothing in the input identifies the field, and the
compiler reports a diagnostic rather than guessing. The last two are fixable and deliberately
left for the next round so that this set's number stays uncontaminated.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 35.5 KiB (budget 39.1 KiB) |
| Weights (int6 text) | 37,775 chars |
| Import + weight decode (Node 24, Xeon 2.9 GHz) | ~5.2 ms |
| First parse (JIT warm-up) | ~13.5 ms |
| Warm CPU parse, 14-token phrase | ~0.63 ms |
| CPU batch, 1000 phrases | ~590 ms |
| WebGPU | canonical scan kernel verified against the CPU path (max |Δ| 1.5e-5 over 27 phrases, one dispatch) on Mesa llvmpipe via wgpu-py; not timed on real hardware |

## Limitations and intended use
Intended for search bars and view builders where the app validates the spec before running
it. Outputs are probabilistic; treat the spec as a proposal and show diagnostics to the user.
Known gaps: comparatives and superlatives whose adjective is not a field word
(`longest books first` with no `long` alias), numbers bound only by a unit noun that names no
field (`under 30 minutes`), unquoted free-text values (`containing catan`), verb-phrase
negation (`games i do not own`), `before`/`after` on a year-like numeric field, clause-level
`or`, negated `contains` and future-facing negated date ranges, `top N` inferring a sort from
an aggregate, bare time phrases occasionally dropped after an enum clause, English only.

## Checkpoint
- Promoted: `runs/default` — 2026-09-18, seed 0, 160,000 samples, 14 epochs (best: 13),
  batch 128, AdamW lr 3e-3 one-cycle, QAT from epoch 1, 2 CPU threads, 738 s wall clock
- Training command: `pnpm train` (`uv run python -m gpu_view.train`), export with `pnpm export`
