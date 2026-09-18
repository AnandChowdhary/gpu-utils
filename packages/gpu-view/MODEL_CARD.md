# Model card: gpu-view

## Task
Natural language to table view specs: filter, sort, group, aggregate, limit, chart. Input is
a search-bar phrase plus the caller's schema; output is a typed spec with character spans and
diagnostics. The model is schema-blind: it tags tokens with roles and clause boundaries using
anonymous schema-membership features, and a deterministic compiler resolves everything else.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime`), whitespace dropped before the model
- Features per token (639 rows, up to 28 active): shape, length bucket, word hash (128),
  consonant-skeleton hash (64), closed task lexicon (365 words + none), 10 flags, and the
  schema-membership rows: field kind / begin-inside / match quality (exact, stem, prefix, typo)
  / alias, enum match / begin-inside / unique owner / owned by nearest preceding or following
  field, kind of and distance to the nearest field match before and after, relative position
- Embedding: 32 dims, summed over active rows
- Sequence mixing: 5-tap depthwise convolution, sigmoid gate + tanh candidate, forward and
  backward affine scans `h[t] = a[t]·h[t−1] + b[t]` (Hillis-Steele prefix scan in training,
  per-channel walk in WGSL), combine, mean-pooled gated global context
- Head: 16-unit gate, 64-unit tanh layer, 14 role logits + 1 clause-boundary logit; argmax decode
- Roles: `O FIELD OP VALUE TIME_VALUE CONJ NEG SORT_FIELD SORT_DIR GROUP_FIELD AGG_FN AGG_FIELD LIMIT CHART`
- Parameters: 33,087 (20,448 embedding + 12,639 backbone/head)
- Quantization: int6 symmetric per-tensor, quantization-aware training from epoch 2 (straight-through)

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
spans included for the generated sets, ignored for the hand-written set (specs were written
without offsets). Numbers are from the promoted checkpoint on the CPU path (`pnpm test`).

| Set | Size | Metric | Score |
|---|---|---|---|
| in-domain (training domains, fresh seed) | 200 | spec exact match, with spans | __INDOMAIN__ |
| held-out (four unseen domains, disjoint vocabulary) | 400 | spec exact match, with spans | __TRANSFER__ |
| held-out | 2,000 | token role accuracy / boundary accuracy | __TOKACC__ / __BNDACC__ |
| unfamiliar (65 hand-written phrases, 4 schemas the generator never saw) | 65 | spec exact match | __UNFAMILIAR__ |

The unfamiliar set (`eval/unfamiliar.json`: podcasts, wine cellar, repositories, greenhouse)
was written before evaluation and was not used for tuning. Failure analysis:

__UNFAMILIAR_ANALYSIS__

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 30.1 KiB (budget 39.1 KiB) |
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
  AdamW lr 3e-3 one-cycle, QAT from epoch 2, 2 CPU threads, __TRAIN_TIME__
- Training command: `pnpm train` (`uv run python -m gpu_view.train`), export with `pnpm export`
