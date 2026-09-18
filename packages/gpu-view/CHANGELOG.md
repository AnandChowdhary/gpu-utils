# gpu-view

## 0.2.0

### Minor Changes

- [#13](https://github.com/AnandChowdhary/gpu-utils/pull/13) [`e01c5bf`](https://github.com/AnandChowdhary/gpu-utils/commit/e01c5bfa6224a64023a1a4b0673c6bf77d9037cd) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - gpu-view v2: moved onto the shared scan model family (canonical WGSL kernel, no package
  kernel) and widened the synthetic generator's coverage categories. New in the featurizer and
  compiler: a `primary` flag on schema fields for date disambiguation, comparative and
  superlative adjectives resolved through field aliases (`cheapest first`, `taller than 50 cm`),
  negation prefixes glued onto a boolean field word (`unarchived`, `nonbillable`, `inactive`),
  unit nouns after a value (`longer than 60 minutes`), unit suffixes on numbers, full dates and
  seasons (`september 10 2026`, `last winter`), year-like numeric fields with polarity bounds
  (`vintage 2019 or older`), `top N` stranded before its sort field, enum values that override a
  carrier noun, and a spec-normalisation pass that merges constraints split across clause
  boundaries. Adds two further hand-written unfamiliar evaluation sets and keeps reporting the
  first, which is now labelled contaminated.

## 0.1.0

### Minor Changes

- [#4](https://github.com/AnandChowdhary/gpu-utils/pull/4) [`3d7d0e4`](https://github.com/AnandChowdhary/gpu-utils/commit/3d7d0e473c2ae3e9e484a4a98171840932d32205) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-view: natural language to table view specs (filter, sort, group, aggregate, limit, chart). A 33K-parameter schema-blind affine-scan tagger that runs on WebGPU or the CPU, with a deterministic TypeScript compiler that resolves fields, dates and numbers and reports diagnostics instead of guessing.
