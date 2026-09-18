# gpu-cite

## 0.1.1

### Patch Changes

- [#11](https://github.com/AnandChowdhary/gpu-utils/pull/11) [`815f84b`](https://github.com/AnandChowdhary/gpu-utils/commit/815f84b88c36b5ee4028e16df175dc82dbf28b4f) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Migrate to the shared `ScanTagger` model family from `@gpu-utils/runtime` / `gpu_utils_training`, retrained. The name-part head now lives in three extra tag columns, the document type in the pooled head, and the CRF transition matrix is an extra exported tensor used only by the decoder; the CPU path uses the runtime's `scanTaggerForward`, the WebGPU path the canonical `scan_tagger.wgsl` kernel, and the package no longer ships its own shader or forward pass.
  
  Measured on the unchanged evaluation sets, the shared family is slightly worse in-distribution and better out of it: held-out synthetic micro-F1 0.990 → 0.987 and exact match 0.702 → 0.680, anystyle micro-F1 0.781 → 0.776, but the hand-written unfamiliar set goes 0.753 → 0.770 micro-F1 / 0.234 → 0.299 exact match and GROBID 0.713 → 0.732 / 0.239 → 0.297. The package shrinks from 54.1 KiB to 52.3 KiB Brotli at 76,747 parameters. See MODEL_CARD.md for the side-by-side table.

## 0.1.0

### Minor Changes

- [#2](https://github.com/AnandChowdhary/gpu-utils/pull/2) [`8f4caad`](https://github.com/AnandChowdhary/gpu-utils/commit/8f4caad9969064e811e16ecb7f9bbf980d201ba0) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Add `gpu-cite`: parse freeform citation and reference strings (APA, MLA, Chicago, IEEE, Vancouver, Harvard, Nature, ACM, arXiv listings, messy copy-paste) into structured bibliographic fields. A 76K-parameter int6 bidirectional affine-scan BIO tagger with a name-part head and a document-type head, deterministic DOI/arXiv/URL extraction, `parse()` and `parseMany()` with UTF-16 spans, CPU reference path and WebGPU kernels.
