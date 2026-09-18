# gpu-cite

Parse freeform citation and reference strings into structured bibliographic fields.

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-cite
```

```ts
import { parse, parseMany } from "gpu-cite";

const record = await parse(
  "Hinton, G., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. " +
    "Neural Computation, 18(7), 1527–1554. https://doi.org/10.1162/neco.2006.18.7.1527",
);
// {
//   type: "article",
//   authors: [{ given: "G.", family: "Hinton", span: [0, 9] }, { given: "S.", family: "Osindero", span: [12, 23] }, ...],
//   title: "A fast learning algorithm for deep belief nets",
//   container: "Neural Computation",
//   year: 2006, volume: "18", issue: "7", pages: { from: "1527", to: "1554" },
//   doi: "10.1162/neco.2006.18.7.1527",
//   spans: { authors: [0, 37], year: [40, 44], title: [47, 93], container: [95, 113], volume: [115, 117], ... },
//   diagnostics: { confidence: 0.99, typeConfidence: 1.0, etAl: false, warnings: [], tags: [...], tokens: [...] },
// }

// Many references, one per line; every span is an offset into the original text.
const records = await parseMany(bibliographyText);
```

Any style works as input: APA, MLA, Chicago (author-date and notes), IEEE, Vancouver, AMA,
Harvard, Nature, ACM, Elsevier/Springer numbered styles, BibTeX `plain`, arXiv listings,
Wikipedia citations, German `Hrsg.`/`S.` forms, and messy copy-paste with missing spaces,
markdown italics or `[CrossRef]` leftovers. Output types are `article`, `book`, `chapter`,
`conference`, `thesis`, `report`, `web`, `preprint` or `unknown` (when the type head is
below 50% confidence).

## How it works

1. **Tokens.** `tokenize()` from `@gpu-utils/runtime` splits the string into runs of letters,
   digits, spaces and single punctuation marks. Each token gets 12 sparse feature ids: a word
   hash, a consonant-skeleton hash, its shape, first/last character, length, relative position
   and up to five flags (year-like, initial, acronym, cue-word groups such as *pp.*, *vol.*,
   *eds.*, *Proceedings*, *University*, months, publisher words, and "inside a DOI/URL/arXiv
   regex match"). No learned vocabulary; the featurizer is byte-for-byte identical in
   TypeScript and Python.
2. **Model.** Summed 32-dim embeddings → bidirectional gated affine scan (32 units per
   direction) → depthwise conv (k=3) with residual → second bidirectional scan → mean/max
   pooled context. A two-layer head emits one BIO tag per token over 15 roles (AUTHOR, TITLE,
   CONTAINER, YEAR, VOLUME, ISSUE, PAGES, PUBLISHER, LOCATION, EDITION, DOI, ARXIV, URL,
   ACCESSED, EDITOR), a 3-way name-part label (given / family / other) and, from the pooled
   vector, the document type. 76,411 parameters, int6, trained with a linear-chain CRF.
3. **Compiler.** Viterbi over the learned CRF transitions with hard BIO constraints runs on
   the CPU. A deterministic TypeScript compiler (`src/decode.ts`) turns entities into typed
   values: trims quotes and brackets, strips *vol./no./pp./ed.* cues, parses page ranges
   (`1477-81` → `1477`–`1481`), splits names with the name-part head, and locates DOIs, arXiv
   ids and URLs with regexes so identifiers are never hallucinated by the model. Every field
   comes with a UTF-16 span into the input.

The WebGPU path (`src/shader.wgsl`) runs the same network over a whole batch of references in
one command buffer; the scans are Hillis–Steele parallel prefix scans over affine maps. Under
`backend: "auto"` inputs below 256 tokens use the CPU reference path, which is also the
fallback when WebGPU is unavailable.

## Size and speed

| Measure | Value |
|---|---|
| Parameters | 76,411 (int6) |
| Package (min + Brotli, weights included) | 54.1 KiB (budget 78.1 KiB) |
| Cold start (import, decode weights, first parse), Node 24 CPU | 29 ms |
| Warm `parse()` of one reference, CPU | ~4 ms |
| Warm `parseMany()` of 1 KB / 7 references, CPU | ~22 ms |

The 80,000-byte budget (instead of the 40 KB default) is justified by the 15-role tag set:
the summed-embedding table alone is 47K of the 76K parameters and each parameter is one
Brotli-compressed character.

## Limitations

- Trained on synthetic renderings of CrossRef/arXiv metadata. On real, hand-labelled corpora
  it is a **field-boundary suggester, not a ground truth**: micro-F1 0.78 on anystyle's core
  set, 0.71 on GROBID's citation corpus, 0.75 on our hand-written unfamiliar set; full-record
  exact match on real data is 23–33%. See [MODEL_CARD.md](./MODEL_CARD.md).
- Weak on: legal citations, patents, standards, journal-abbreviation-only physics references
  with `ibid.`, non-Latin scripts (Cyrillic, CJK), references where the container and the
  location are glued (`Journal 12, London`), `in press`/`forthcoming` dates (no `year` is
  returned; a warning is added), HTML leftovers (`<i>Nature</i>`).
- `parseMany()` assumes one reference per line; wrapped references must be joined first.
- `year` is always a number; the full date text is available through `spans.year`.
  `accessed` is returned as written.
- Only DOIs, arXiv ids and URLs are validated. Everything else is probabilistic; check
  `diagnostics.confidence`, `diagnostics.typeConfidence` and `diagnostics.warnings`.

## Training

```bash
cd packages/gpu-cite/training
uv sync
uv run python -m gpu_cite.sources   # download CrossRef/arXiv metadata + anystyle/GROBID eval corpora
uv run python -m gpu_cite.data      # render 120K labelled references in 19 styles
uv run python -m gpu_cite.train     # 4 epochs, QAT int6, ~13 min on 2 CPU threads
uv run python -m gpu_cite.evaluate  # held-out, unfamiliar, anystyle, GROBID
uv run python -m gpu_cite.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
