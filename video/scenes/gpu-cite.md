# How gpu-cite turns a reference string into fields

## Overview

- **Topic**: Parsing a freeform citation ("Smith, J., & Doe, A. (2019). A study of things. Journal of Stuff, 12(3), 45–67.") into authors, title, container, year, volume, issue, pages, DOI.
- **Hook**: Every reference list is a different dialect. A 76K-parameter tagger reads the punctuation and cue words, not a dictionary.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 50–60 seconds, silent, 16:9.
- **Key Insight**: Field *boundaries* are the hard part. The model only decides where each field starts and stops (a BIO tag per token); a deterministic compiler turns those tags into typed values, and DOIs/arXiv ids/URLs are matched by regex, never guessed.

## Narrative Arc

Start with one APA reference. Split it into character-class tokens. Show each token
becoming a handful of sparse hashed features (word, skeleton, shape, cue-word flags).
Show the two bidirectional gated scans sweeping left-to-right and right-to-left over the
same cells, then the tag per token lighting up in a semantic colour. Finally the cells
regroup into the typed record with the name-part split (given/family) and the regex-locked
DOI. Hold on the record and the size line.

---

## Scene 1: A reference

**Duration**: ~6 seconds

**Purpose**: Establish the input as one familiar string.

### Visual Elements

- Caption "A reference, any style".
- The reference rendered as one line of `mono()` text, white.
- Muted footnote: "APA · MLA · IEEE · Vancouver · Harvard · Nature · messy copy-paste".

### Technical Notes

- Use `Text` with the bundled font, not `Code`.

---

## Scene 2: Tokens

**Duration**: ~7 seconds

**Purpose**: Show the mechanical tokenizer: runs of letters, digits, spaces and single punctuation marks.

### Visual Elements

- Caption "Split into character-class tokens".
- The line transforms (`ReplacementTransform`) into a row of `TokenCell`s; spaces muted.
- Footnote: "no vocabulary, no learned tokenizer".

---

## Scene 3: Sparse features

**Duration**: ~8 seconds

**Purpose**: Each token becomes a few hashed ids and cue-word flags.

### Visual Elements

- Caption "12 sparse feature ids per token".
- `activation_strip()` under each cell (accent colour = learned state).
- Small muted labels under three cells: `word·skeleton·shape`, `flag: YEARLIKE`, `flag: IN_DOI`.

### Accuracy notes

- Flags shown must be real feature names from `src/features.ts`.

---

## Scene 4: Two scans

**Duration**: ~9 seconds

**Purpose**: The trunk is the shared scan family: two bidirectional gated affine scans, each followed by a residual 5-tap depthwise conv.

### Visual Elements

- Caption "Bidirectional gated scans".
- An accent-coloured window sweeps left→right over the cells, then a second sweeps right→left; a thin accent bar connects all cells (pooled context).
- Footnote: "h = a·h_prev + (1−a)·tanh(u), as a parallel prefix scan on the GPU".

---

## Scene 5: Tags

**Duration**: ~8 seconds

**Purpose**: The model's only output: a BIO tag per token, plus a document type.

### Visual Elements

- Caption "One tag per token".
- Cells recolour by role (semantic colours appear here for the first time): AUTHOR, YEAR, TITLE, CONTAINER, VOLUME, ISSUE, PAGES, DOI. Punctuation stays white/muted (O).
- A small pill "type: article" appears at the right.

---

## Scene 6: Compiler

**Duration**: ~10 seconds

**Purpose**: Deterministic TypeScript turns tags into typed values; regex locks identifiers.

### Visual Elements

- Caption "Deterministic compiler".
- Cells regroup (`ReplacementTransform`) into a key/value block:
  `authors: [{given: "J.", family: "Smith"}, {given: "A.", family: "Doe"}]`, `year: 2019`, `title`, `container`, `volume: "12"`, `issue: "3"`, `pages: {from: "45", to: "67"}`, `doi: "10.1000/xyz123"`.
- Footnote: "DOI, arXiv id and URL come from regex, never from the model".

---

## Scene 7: On device

**Duration**: ~6 seconds + 3 s hold

**Purpose**: The payoff: tiny, local, no server.

### Visual Elements

- Caption "Runs on your GPU. No server."
- Three numbers from MODEL_CARD.md: parameters, Brotli size, warm latency.
- End on a 3 s hold.

---

## Recurring Visual Motifs

- Token cells (`TokenCell`) persist from decomposition to final labels.
- Accent `#3B82F6` marks learned state only (feature strips, scan window, pooled bar).

## Color Palette

| Role | Hex | Usage |
|---|---|---|
| Background | #000000 | frame |
| Text | #F5F5F5 | primary |
| Muted | #999999 | footnotes, whitespace glyphs |
| Accent | #3B82F6 | activations, scan windows |
| AUTHOR | #F59E0B | model output only |
| TITLE | #10B981 | model output only |
| CONTAINER | #8B5CF6 | model output only |
| YEAR / VOLUME / ISSUE / PAGES | #EC4899 | model output only |
| DOI | #06B6D4 | model output only |

## Accuracy Guardrails

- The animation must not imply the model sees field names, a dictionary of journals, or that
  it generates text. It only tags tokens.
- The DOI must be shown as matched by regex, not by the model.
- Every number shown (parameters, Brotli size, latency) matches MODEL_CARD.md.
- Do not imply the output is deterministic for messy inputs; the footnote in scene 5 says "probabilistic tags".

## Reference Material

- `packages/gpu-cite/src/features.ts`
- `packages/gpu-cite/src/decode.ts`
- `packages/gpu-cite/MODEL_CARD.md`
