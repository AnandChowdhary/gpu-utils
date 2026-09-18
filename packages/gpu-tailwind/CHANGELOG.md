# gpu-tailwind

## 0.1.0

### Minor Changes

- [#8](https://github.com/AnandChowdhary/gpu-utils/pull/8) [`84f10e9`](https://github.com/AnandChowdhary/gpu-utils/commit/84f10e9bf9ea4db1e884f43d25786b5e8f37f2d1) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Add gpu-tailwind: natural language to Tailwind CSS v4 utility classes. A ~45K-parameter
  bidirectional affine-scan tagger labels tokens as property / value / variant / separator /
  negation and scores segment boundaries; a deterministic compiler maps (property, value)
  pairs to classes through a table compiled from the Tailwind v4 default theme, applies
  responsive and state variants per phrase segment, and validates every class against the
  compiled vocabulary. Runs on WebGPU with a CPU reference path.
