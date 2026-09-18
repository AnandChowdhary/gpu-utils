# Model card: gpu-cite

## Task
Parse freeform citation and reference strings into structured bibliographic fields: a BIO
span tagger over 15 roles plus a name-part head and a document-type head, followed by a
deterministic compiler.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime` `tokenize`); 12 sparse hashed feature
  ids per token (word 1024, consonant skeleton 256, shape 8, first/last char 64+64, length 16,
  position 8, up to five of 38 lexicon/regex flags). Table: 1,479 rows × 32 dims.
- Embedding: 32 dims, summed.
- Sequence mixing: bidirectional gated affine scan `h = a·h_prev + (1−a)·b`, 32 units per
  direction → depthwise conv k=3 + ReLU residual → second bidirectional scan (32 per
  direction) → mean/max pooling over tokens.
- Heads: token head `relu([h2 | y | mean] W1 + b1)` (192→48) → 31 BIO tags and 3 name parts;
  type head `relu([mean | max] Wc + bc) Wd` (128→32→8).
- Decoder: learned CRF transitions (31×31) + hard BIO constraints, Viterbi on the CPU, then
  `src/decode.ts` compiles entities into typed fields; DOI / arXiv / URL from regexes.
- Parameters: 76,411 (embedding 47,328; scans 12,544; conv 192; heads 13,386; type head
  4,392; transitions 961).
- Quantization: int6 symmetric per-tensor, quantization-aware training (fake-quant with a
  straight-through estimator from epoch 1). The exported `weights.txt` is decoded back and
  used to compute the parity fixtures, so `model/fixtures.json` is exactly what the runtime
  computes.

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

### Held-out synthetic (4,000 references, disjoint metadata)

| Field | P | R | F1 | Support |
|---|---|---|---|---|
| AUTHOR | 0.991 | 0.990 | 0.991 | 3976 |
| TITLE | 0.973 | 0.978 | 0.975 | 3859 |
| CONTAINER | 0.977 | 0.983 | 0.980 | 2906 |
| YEAR | 0.994 | 1.000 | 0.997 | 4174 |
| VOLUME | 0.998 | 0.996 | 0.997 | 1899 |
| ISSUE | 0.999 | 0.998 | 0.999 | 1217 |
| PAGES | 0.995 | 0.994 | 0.994 | 2211 |
| PUBLISHER | 0.982 | 0.989 | 0.985 | 970 |
| LOCATION | 0.995 | 0.995 | 0.995 | 421 |
| EDITION | 1.000 | 0.986 | 0.993 | 72 |
| DOI | 1.000 | 1.000 | 1.000 | 774 |
| ARXIV | 0.995 | 0.998 | 0.996 | 393 |
| URL | 1.000 | 1.000 | 1.000 | 415 |
| ACCESSED | 1.000 | 1.000 | 1.000 | 179 |
| EDITOR | 0.984 | 0.988 | 0.986 | 319 |
| **micro** | | | **0.990** | |

Token accuracy 0.997 · type accuracy 0.978 · author-entity F1 0.992 · full-record exact match
**0.702** (0.715 ignoring the type). Exact match by style: apa 0.89, harvard 0.87, ieee 0.82,
springer 0.82, elsevier 0.81, cse 0.80, nature 0.77, physics 0.75, ama 0.74, vancouver 0.73,
german 0.69, wikipedia 0.69, mla 0.66, plain 0.61, chicago_ad 0.56, chicago_note 0.52,
arxiv_listing 0.47, messy 0.42, acm 0.41.

### Unfamiliar, hand-written (77 references; styles the renderer does not produce)
BibTeX source, ABNT, GOST (Cyrillic), Chinese and Japanese references, Bluebook and OSCOLA
law citations, a patent, markdown/HTML leftovers, OCR noise (`I979`, `29l`), hyphenated line
wraps, software/dataset/video/podcast/tweet citations, ISO standards, RFCs, government
reports, French/German/Spanish theses and chapters, `Jr.` and particle surnames, `in press`
and `forthcoming`, `Volume 52, Issue 1, Pages`, `63 #4`, `ibid.`, tab-separated fields, a
bare URL and a bare title.

| Field | P | R | F1 | Support |
|---|---|---|---|---|
| AUTHOR | 0.913 | 0.900 | 0.906 | 70 |
| TITLE | 0.634 | 0.703 | 0.667 | 74 |
| CONTAINER | 0.571 | 0.696 | 0.627 | 46 |
| YEAR | 0.805 | 0.849 | 0.827 | 73 |
| VOLUME | 0.767 | 0.793 | 0.780 | 29 |
| ISSUE | 0.609 | 0.609 | 0.609 | 23 |
| PAGES | 0.738 | 0.912 | 0.816 | 34 |
| PUBLISHER | 0.455 | 0.652 | 0.536 | 23 |
| LOCATION | 0.733 | 0.550 | 0.629 | 20 |
| EDITION | 1.000 | 0.500 | 0.667 | 6 |
| DOI | 0.917 | 0.917 | 0.917 | 12 |
| ARXIV | 1.000 | 1.000 | 1.000 | 2 |
| URL | 0.875 | 0.875 | 0.875 | 16 |
| ACCESSED | 1.000 | 1.000 | 1.000 | 4 |
| EDITOR | 1.000 | 1.000 | 1.000 | 1 |
| **micro** | | | **0.753** | |

Type accuracy 0.838 (74 typed cases; the three `unknown` cases are excluded) · author-entity
F1 0.914 · full-record exact match **0.234** (18 / 77). These numbers are for the model's
spans alone; the TypeScript runtime additionally overlays regex matches for DOI / arXiv /
URL, so identifier accuracy in `parse()` is higher than the table (e.g. the model cut
`…/NY.GDP.PCAP.CD` at `NY.GDP`; the regex returns the full URL).

### Real labelled corpora (model spans only, evaluation roles limited to what each corpus annotates)

| Set | Records | micro-F1 | macro-F1 | Exact match | Strong fields (F1) | Weak fields (F1) |
|---|---|---|---|---|---|---|
| anystyle core (BSD-2) | 1,514 | **0.781** | 0.720 | 0.326 | AUTHOR 0.91, YEAR 0.90, PAGES 0.89, DOI 0.88 | VOLUME 0.38 (anystyle puts the issue inside `<volume>`), EDITION 0.47, EDITOR 0.64, PUBLISHER 0.66 |
| GROBID citation corpus (Apache-2.0) | 3,525 | **0.713** | 0.619 | 0.239 | YEAR 0.92, PAGES 0.87, AUTHOR 0.87, DOI 0.80 | ARXIV 0.21 (`arXiv .org:astro -ph/0710.3063` spacing artefacts), URL 0.39, CONTAINER 0.40 (PDF-glued tokens such as `Optics letters28`, `Technol.27`) |

Neither corpus was used for training or tuning.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 54.1 KiB / budget 78.1 KiB (80,000 B) |
| Cold start (import + int6 decode + first parse, Node 24, CPU) | 29 ms |
| Warm `parse()` one reference (~40–80 tokens), CPU | ~4 ms median |
| Warm `parseMany()` 1 KB / 7 references, CPU | ~22 ms median |
| WebGPU | used under `auto` for ≥256 tokens; one command buffer of 11 passes per batch of up to 512 references. Wall-clock not benchmarked on a real GPU (only a llvmpipe software adapter was available); expect dispatch + readback to dominate. |

WGSL parity: `training/tests/test_wgsl.py` runs the real `src/shader.wgsl` through wgpu-py on
a WebGPU adapter (llvmpipe / CPU Vulkan in CI and on the dev box; any GPU elsewhere) with the
same batched buffer layout as `src/gpu.ts` and one bind group per pipeline, as
`runtime/program.ts` does. Against the PyTorch fixtures the worst deviation is 1.9e-5 over
24 references / 1,498 tokens dispatched in one command buffer, and 1.5e-5 on a 341-token
reference that crosses the 256-token scan-chunk boundary (test tolerance 1e-3). The kernels
are also `naga`-validated (parse, type-check, uniformity analysis) and minified by
`wgslender` at build time. The CPU path remains the source of truth and matches PyTorch on
the same 24 fixtures at 1e-4.

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
- The type head is calibrated on synthetic data: 98% on held-out, 84% on the unfamiliar set;
  patents and legal cases come out as `article`.
- Never generates text. Outputs are probabilistic except DOI / arXiv / URL, which are regex
  matches.
- WebGPU latency has only been exercised on a software adapter; real-GPU timings are pending.

## Checkpoint
- Promoted: `training/runs/default` — 2026-09-17, seed 0, epoch 3 of 4 (best held-out exact
  match), 13.1 min on 2 threads, OneCycle AdamW lr 4e-3, batch 128, CRF NLL + 0.5·name-part CE
  + 0.5·type CE, QAT from epoch 1.
- Per-epoch held-out exact match / micro-F1: 0.414 / 0.975 → 0.570 / 0.987 → 0.683 / 0.987 →
  0.702 / 0.990.
- Training command: `pnpm train` (`uv run python -m gpu_cite.train`); export: `pnpm export`.
