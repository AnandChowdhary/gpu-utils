# gpu-paste

## 0.1.1

### Patch Changes

- [#9](https://github.com/AnandChowdhary/gpu-utils/pull/9) [`f0d5473`](https://github.com/AnandChowdhary/gpu-utils/commit/f0d547373f4849153536a460bf23066bb31a1180) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Migrate to the shared model family, retrained. The learned part of gpu-paste is now a
  plain `ScanTagger` from the shared runtime (BIO span head = per-token tags, kind head =
  pooled output): the CPU path uses `scanTaggerForward`, the WebGPU path `runScanTagger` on
  the canonical `scan_tagger.wgsl` kernel (all 512-token windows of a long paste run as one
  batched dispatch), and training/export/evaluation go through `gpu_utils_training`. The
  package's own model layers, fake-quant, training loop, Viterbi tables, `shader.wgsl` and
  hand-written forward pass are gone; rules, detectors, decoder policy, data generator and
  the evaluation sets are unchanged.
  
  Scored against the previous checkpoint on the same sets: held-out kind accuracy
  0.9923 → 0.9930 and span micro-F1 0.9820 → 0.9864; end-to-end span micro-F1 on the 77
  hand-written pastes 72.8 % → 74.3 %; end-to-end kind accuracy 94.8 % → 92.2 % (2 cases,
  inside the 92.2–94.8 % spread of three seeds of the new family). 66,339 → 68,659
  parameters, 50.9 → 53.7 KiB Brotli against a 60 KB budget, and the CPU forward pass is
  ~45 % slower. See MODEL_CARD.md for the full before/after tables.

## 0.1.0

### Minor Changes

- [#1](https://github.com/AnandChowdhary/gpu-utils/pull/1) [`e600415`](https://github.com/AnandChowdhary/gpu-utils/commit/e6004157b31d24a3b67f3d6bf42d8c54b28ab84c) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-paste: understand pasted text on-device. Deterministic detectors
  for JSON, CSV/TSV, HTML, URL, email, phone, datetime, color, UUID, JWT, IP, path, money and
  number; a 66K-parameter int6 bidirectional affine-scan model classifies address / contact /
  prose / list / code / markdown and tags person, company, address, date, money and phone spans
  inside mixed text. CPU reference path plus WGSL kernels, zero runtime dependencies.
