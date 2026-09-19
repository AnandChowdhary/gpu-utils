# gpu-tailwind

## 0.2.0

### Minor Changes

- [#15](https://github.com/AnandChowdhary/gpu-utils/pull/15) [`9fb9ce3`](https://github.com/AnandChowdhary/gpu-utils/commit/9fb9ce3e7cc5fcd17f3e2331d3a5cff795ba1ef8) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - gpu-tailwind v2: move the model onto the shared `ScanTagger` family and the canonical
  `scan_tagger.wgsl` kernel from `@gpu-utils/runtime` (the package no longer ships its own
  shader), widen the synthetic generator to cover gradients, colour alpha, shade wording,
  two-colour "X on a Y background" units, number words and heavier casing/typo noise, and
  teach the compiler leading-variant scoping. Adds a second hand-written unfamiliar
  evaluation set and makes the held-out split disjoint from the training set.

## 0.1.0

### Minor Changes

- [#8](https://github.com/AnandChowdhary/gpu-utils/pull/8) [`84f10e9`](https://github.com/AnandChowdhary/gpu-utils/commit/84f10e9bf9ea4db1e884f43d25786b5e8f37f2d1) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Add gpu-tailwind: natural language to Tailwind CSS v4 utility classes. A ~45K-parameter
  bidirectional affine-scan tagger labels tokens as property / value / variant / separator /
  negation and scores segment boundaries; a deterministic compiler maps (property, value)
  pairs to classes through a table compiled from the Tailwind v4 default theme, applies
  responsive and state variants per phrase segment, and validates every class against the
  compiled vocabulary. Runs on WebGPU with a CPU reference path.
