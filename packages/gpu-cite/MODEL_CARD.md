# Model card: gpu-cite

## Task
Parse freeform citation and reference strings into structured bibliographic fields: a BIO
span tagger over 15 roles plus a name-part head and a document-type head (all on the shared
scan model family), followed by a deterministic compiler.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime` `tokenize`); 12 sparse hashed feature
  ids per token (word 1024, consonant skeleton 256, shape 8, first/last char 64+64, length 16,
  position 8, up to five of 38 lexicon/regex flags). Table: 1,479 rows × 32 dims; rows are
  padded with id 0 (`paddingId: 0`, row 0 is never read).
- Family: `ScanTagger(feature_rows=1479, hidden=32, tags=34, pooled_out=8, scan_layers=2)` from
  `gpu_utils_training.models` — the shared scan family, unchanged: summed embeddings → two
  layers of bidirectional gated affine scan `h = a·h_prev + (1−a)·tanh(u)` (32 units per
  direction) each followed by a residual 5-tap depthwise conv → mean-pooled context →
  `relu([x_t | ctx] W_head + b)` (128→64) → tag columns; `relu(ctx W_p1 + b) W_p2` (64→64→8)
  → pooled columns.
- Heads, mapped onto the family's escape hatches: the 34 tag columns are 31 BIO tags over 15
  roles followed by 3 name-part labels (O / GIVEN / FAMILY); the 8 pooled columns are the
  document type. The decoder splits them.
- Decoder: learned CRF transitions (`trans`, 31×31, an extra exported tensor read only by the
  decoders) + the runtime's hard BIO constraints (`bioTransitions`, `bioStartMask`), Viterbi on
  the CPU, then `src/decode.ts` compiles entities into typed fields; DOI / arXiv / URL from
  regexes.
- Runtime: CPU path `scanTaggerForward`, WebGPU path `runScanTagger` on the canonical
  `scan_tagger.wgsl` kernel — no package-specific forward pass or shader.
- Parameters: 76,747 (embedding 47,328; scan layers 4,224 + 8,320; convs 384 + 384; head 8,256;
  tags 2,210; pooled head 4,680; transitions 961). Before the migration: 76,411.
- Quantization: int6 symmetric per-tensor, quantization-aware training (`gpu_utils_training.qat`,
  straight-through estimator from epoch 1). `model/fixtures.json` is computed from the decoded
  `weights.txt`, so it is exactly what the runtime computes.

## Training data
All downloaded at training time by `training/gpu_cite/sources.py`; nothing over 2 MB is
committed.

| Source | Use | Size | License |
|---|---|---|---|
| CrossRef REST API `/works?sample=100` (60 requests) | structured metadata for the synthetic renderer | 4,412 records (3,486 articles, 440 chapters, 264 proceedings papers, 144 posted-content, 49 books, 29 reports) | CrossRef metadata is CC0 / public domain |
| arXiv OAI-PMH `ListRecords` (`arXiv` metadata prefix) | structured metadata for preprints | 1,040 records | CC0 |
| Synthetic renderer (`training/gpu_cite/data.py`) | training + held-out | 120,000 train / 4,000 held-out (held-out records are disjoint at the metadata level) | — |
| anystyle `res/parser/core.xml` | evaluation only | 1,514 references | BSD-2-Clause |
| GROBID `grobid-trainer/resources/dataset/citation` (`*.references*.xml`) | evaluation only | 3,525 references (115 files) | Apache-2.0 |
| Hand-written unfamiliar set (`training/data/unfamiliar.jsonl`) | evaluation only, never tuned on | 77 references | this repository (MIT) |

Not used: CORA and the UMass Citation Field Extraction dataset. Their distribution pages are
login-gated or carry no license text we could verify, so they were excluded.

The renderer produces 19 style families (APA, MLA, Chicago author-date, Chicago notes, IEEE,
Vancouver, AMA, Harvard, Nature, ACM, Elsevier numbered, Springer, BibTeX plain, arXiv
listing / Google Scholar / Semantic Scholar rows, Wikipedia cite, CSE, terse physics, German,
deliberately messy) with per-render variation: initials vs full names (`J. A.`, `J.A.`, `JA`,
`John A.`), `et al.` cut-offs and spellings (`et al`, `and others`, `u. a.`), `and`/`&`/`und`/`y`,
quote styles, sentence/title/upper casing, `12(3)` / `vol. 12, no. 3` / `12:3` / `Bd. 12, H. 3`,
page ranges with `-`/`–`/`—`/`to` and abbreviated end pages, `(n.d.)`, month names, numbering
prefixes, `[CrossRef]`/`PMID` junk, markdown italic leftovers, dropped or doubled spaces and
length-preserving typos. 30–38% of CrossRef articles are re-realised as books, theses,
reports, web pages, chapters or conference papers with synthesized institutions, sites, URLs
and report numbers so all eight types are covered.

## Evaluation
Exact-span metrics after trimming edge punctuation and cue words (`vol.`, `pp.`, `(Eds.)`) on
both sides; AUTHOR/EDITOR are scored as the union span of the name list. "Exact match" is a
whole record: every field span, the document type, and (on the held-out set) the given/family
split of every name token.

### Migration summary: custom model (v1) vs shared scan family

Identical evaluation sets, identical scorer, identical featurizer; only the model and the
runtime changed. v1 is the checkpoint merged to `main` (hand-written `nn.Module`, package-local
`cpu.ts` forward and `shader.wgsl`); v2 is `ScanTagger` from `gpu_utils_training` on the
canonical `scan_tagger.wgsl` kernel.

| Evaluation set | Metric | v1 (custom) | v2 (shared family) | Δ |
|---|---|---|---|---|
| Held-out synthetic (4,000) | micro-F1 | 0.990 | 0.987 | **−0.003** |
| | full-record exact match | 0.702 | 0.680 | **−0.022** |
| | token accuracy | 0.997 | 0.996 | −0.001 |
| | type accuracy | 0.978 | 0.977 | −0.001 |
| | author-entity F1 | 0.992 | 0.988 | −0.004 |
| Unfamiliar, hand-written (77) | micro-F1 | 0.753 | 0.770 | **+0.017** |
| | full-record exact match | 0.234 | 0.299 | **+0.065** |
| | type accuracy | 0.838 | 0.851 | +0.013 |
| | author-entity F1 | 0.914 | 0.931 | +0.017 |
| anystyle core (1,514) | micro-F1 | 0.781 | 0.776 | **−0.005** |
| | macro-F1 | 0.720 | 0.713 | −0.007 |
| | exact match | 0.326 | 0.318 | −0.008 |
| GROBID citations (3,525) | micro-F1 | 0.713 | 0.732 | **+0.019** |
| | macro-F1 | 0.619 | 0.629 | +0.010 |
| | exact match | 0.239 | 0.297 | **+0.058** |
| — | parameters | 76,411 | 76,747 | +336 |
| — | package, min + Brotli | 54.1 KiB | 52.3 KiB | **−1.8 KiB** |

Read plainly: the shared family costs a little in-distribution (synthetic exact match −2.2
points, anystyle micro-F1 −0.5 points) and gains on everything the generator did not produce
(unfamiliar +6.5 points exact match, GROBID +5.8). The trade is a wash on aggregate quality
and a clear win on shipped size and on the amount of code the package owns. The synthetic
regression is the honest cost of the migration and is not hidden by the real-corpus gains:
v2 is measurably worse at the distribution it was trained on.

### Held-out synthetic (4,000 references, disjoint metadata)

| Field | P | R | F1 | Support |
|---|---|---|---|---|
| AUTHOR | 0.988 | 0.987 | 0.988 | 3976 |
| TITLE | 0.968 | 0.974 | 0.971 | 3859 |
| CONTAINER | 0.968 | 0.973 | 0.970 | 2906 |
| YEAR | 0.992 | 1.000 | 0.996 | 4174 |
| VOLUME | 0.997 | 0.994 | 0.996 | 1899 |
| ISSUE | 0.998 | 0.995 | 0.997 | 1217 |
| PAGES | 0.994 | 0.993 | 0.993 | 2211 |
| PUBLISHER | 0.984 | 0.990 | 0.987 | 970 |
| LOCATION | 1.000 | 0.995 | 0.998 | 421 |
| EDITION | 1.000 | 1.000 | 1.000 | 72 |
| DOI | 1.000 | 1.000 | 1.000 | 774 |
| ARXIV | 1.000 | 0.998 | 0.999 | 393 |
| URL | 1.000 | 1.000 | 1.000 | 415 |
| ACCESSED | 1.000 | 1.000 | 1.000 | 179 |
| EDITOR | 0.969 | 0.969 | 0.969 | 319 |
| **micro** | 0.986 | 0.988 | **0.987** | 23785 |

Token accuracy 0.996 · type accuracy 0.977 · author-entity F1 0.988 · macro-F1 0.991 ·
full-record exact match **0.680** (0.692 ignoring the type). Exact match by style: apa 0.86,
elsevier 0.85, harvard 0.81, ieee 0.80, springer 0.76, cse 0.75, physics 0.73, vancouver 0.73,
nature 0.72, ama 0.69, german 0.66, mla 0.65, wikipedia 0.62, plain 0.59, chicago_ad 0.59,
acm 0.50, arxiv_listing 0.42, chicago_note 0.41, messy 0.40.

### Unfamiliar, hand-written (77 references; styles the renderer does not produce)
BibTeX source, ABNT, GOST (Cyrillic), Chinese and Japanese references, Bluebook and OSCOLA
law citations, a patent, markdown/HTML leftovers, OCR noise (`I979`, `29l`), hyphenated line
wraps, software/dataset/video/podcast/tweet citations, ISO standards, RFCs, government
reports, French/German/Spanish theses and chapters, `Jr.` and particle surnames, `in press`
and `forthcoming`, `Volume 52, Issue 1, Pages`, `63 #4`, `ibid.`, tab-separated fields, a
bare URL and a bare title.

| Field | P | R | F1 | Support |
|---|---|---|---|---|
| AUTHOR | 0.886 | 0.886 | 0.886 | 70 |
| TITLE | 0.651 | 0.730 | 0.688 | 74 |
| CONTAINER | 0.600 | 0.783 | 0.679 | 46 |
| YEAR | 0.849 | 0.849 | 0.849 | 73 |
| VOLUME | 0.767 | 0.793 | 0.780 | 29 |
| ISSUE | 0.700 | 0.609 | 0.651 | 23 |
| PAGES | 0.795 | 0.912 | 0.849 | 34 |
| PUBLISHER | 0.517 | 0.652 | 0.577 | 23 |
| LOCATION | 0.588 | 0.500 | 0.540 | 20 |
| EDITION | 1.000 | 0.500 | 0.667 | 6 |
| DOI | 0.846 | 0.917 | 0.880 | 12 |
| ARXIV | 1.000 | 1.000 | 1.000 | 2 |
| URL | 1.000 | 1.000 | 1.000 | 16 |
| ACCESSED | 1.000 | 1.000 | 1.000 | 4 |
| EDITOR | 1.000 | 1.000 | 1.000 | 1 |
| **micro** | 0.748 | 0.794 | **0.770** | 433 |

Type accuracy 0.851 (74 typed cases; the three `unknown` cases are excluded) · author-entity
F1 0.931 · full-record exact match **0.299** (23 / 77). These numbers are for the model's
spans alone; the TypeScript runtime additionally overlays regex matches for DOI / arXiv /
URL, so identifier accuracy in `parse()` is higher than the table.

### Real labelled corpora (model spans only, evaluation roles limited to what each corpus annotates)

| Set | Records | micro-F1 | macro-F1 | Exact match | Strong fields (F1) | Weak fields (F1) |
|---|---|---|---|---|---|---|
| anystyle core (BSD-2) | 1,514 | **0.776** | 0.713 | 0.318 | YEAR 0.92, AUTHOR 0.90, PAGES 0.88, DOI 0.88 | VOLUME 0.39 (anystyle puts the issue inside `<volume>`), EDITION 0.48, EDITOR 0.59, LOCATION 0.64 |
| GROBID citation corpus (Apache-2.0) | 3,525 | **0.732** | 0.629 | 0.297 | YEAR 0.93, AUTHOR 0.88, PAGES 0.85, TITLE 0.73 | ARXIV 0.17 (`arXiv .org:astro -ph/0710.3063` spacing artefacts), URL 0.42, CONTAINER 0.47 (PDF-glued tokens such as `Optics letters28`, `Technol.27`) |

Neither corpus was used for training or tuning, and neither eval set changed in this migration.

## Size and latency
| Measure | Value | v1 |
|---|---|---|
| Package (min + Brotli, incl. weights) | 52.3 KiB / budget 78.1 KiB (80,000 B) | 54.1 KiB |
| Cold start (import + int6 decode + first parse, Node 24, CPU) | 25 ms | 29 ms |
| Warm `parse()` one reference (~40–80 tokens), CPU | ~5 ms median | ~4 ms |
| Warm `parseMany()` 1.2 KB / 7 references, CPU | ~35 ms median | ~22 ms |
| WebGPU | used under `auto` for ≥256 tokens; one command buffer per batch of up to 512 references on the canonical `scan_tagger.wgsl`. Wall-clock not benchmarked on a real GPU (only a Mesa lavapipe software adapter was available); expect dispatch + readback to dominate. | |

The CPU path is slower than v1 (the family runs two 5-tap depthwise convs and a wider
128→64 head, where v1 had one 3-tap conv and a 192→48 head). v1's latency row was measured on
a different machine, so treat the CPU deltas as indicative rather than a controlled comparison;
the size and accuracy numbers above *are* controlled.

WGSL parity: `training/tests/test_wgsl.py` runs the canonical `packages/runtime/src/wgsl/scan_tagger.wgsl`
through wgpu-py on a real WebGPU adapter (Mesa lavapipe / CPU Vulkan in CI and on the dev box;
any GPU elsewhere) via `gpu_utils_training.kernels`, with the same batched buffer layout and
one bind group per pipeline that `src/gpu.ts` uses through `runScanTagger`. Against the
PyTorch fixtures the worst deviation is **1.5e-5** over 24 references dispatched in one command
buffer, and the same order on a 300+ token reference that crosses the scan-chunk boundary
(test tolerance 1e-4). This package no longer ships a shader of its own. The CPU path remains
the source of truth and matches PyTorch on the same 24 fixtures at 1e-4.

## Limitations and intended use
- Real-world field boundaries are right roughly 70–80% of the time and whole records
  roughly a quarter to a third of the time. Use it to pre-fill forms, suggest spans or
  bucket references; verify before storing.
- Systematic failure modes seen on the unfamiliar set: a leading `(Host)`, `[Video]`,
  `[Data set]`, `(ISO Standard No. …)` gets absorbed into the title; `Jr.` is dropped from
  the name; `#4` and `Volume 52, Issue 1` cue forms are only partly stripped; two-part
  locations (`Scotts Valley, CA`) are split between location and publisher; `ibid.` repeats
  create duplicate spans; `forthcoming`/`in press` are not tagged as dates; Cyrillic and
  CJK tokens are out of distribution.
- The type head is calibrated on synthetic data: 98% on held-out, 85% on the unfamiliar set;
  patents and legal cases come out as `article`.
- Never generates text. Outputs are probabilistic except DOI / arXiv / URL, which are regex
  matches.
- WebGPU latency has only been exercised on a software adapter; real-GPU timings are pending.
- Against v1 this checkpoint is slightly worse on the synthetic distribution it was trained on
  (exact match 0.680 vs 0.702) and slightly better on everything else. If you depend on
  synthetic-style references specifically, that regression is real.

## Checkpoint
- Promoted: `training/runs/default` — 2026-09-18, seed 0, epoch 4 of 5 (best held-out exact
  match), 15.6 min on 2 threads, `gpu_utils_training.loop.train` (AdamW, warm-up + cosine,
  peak lr 3e-3, batch 128), CRF NLL + 0.5·name-part CE + 0.5·type CE, int6 QAT from epoch 1.
  These are the parser defaults, so a bare `pnpm train` reproduces this checkpoint.
- Per-epoch held-out exact match / micro-F1: 0.378 / 0.971 → 0.542 / 0.981 → 0.632 / 0.984 →
  0.676 / 0.986 → 0.680 / 0.987.
- Training command: `pnpm train` (`uv run python -m gpu_cite.train`); export: `pnpm export`
  (`uv run python -m gpu_cite.export --run default`).
- The previous, hand-written v1 checkpoint is the one merged to `main` before this migration;
  its numbers are reproduced in the migration summary above for comparison.
