# gpu-log

## 0.2.0

### Minor Changes

- [#14](https://github.com/AnandChowdhary/gpu-utils/pull/14) [`04e6067`](https://github.com/AnandChowdhary/gpu-utils/commit/04e60675c69e29e5e2d77f08837b6627050ff327) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - gpu-log v2: the model moves onto the shared conv family (`ConvTagger`, the canonical
  `conv_tagger.wgsl` kernel, no package-specific shader), gains a first-class `host` field
  (hostname, node id or IP), and is retrained on a much wider generator — cluster/supercomputer
  RAS logs, Windows CBS/CSI and Event Viewer text, Proxifier, syslog wrapping a structured app
  line, ten bracket/parenthesis wrappers for source and thread, multi-word sources, quoted and
  nested-brace values, and key=value prose inside messages.
  
  Evaluation is new too: besides the frozen v1 hand-written set (kept for continuity, and
  contaminated for v2 because it drove the widening) and a fresh hand-written set, gpu-log is
  now scored on 16,000 **real** log lines from Loghub, with gold spans derived mechanically
  from Loghub's own structured CSVs rather than from our generator. Two generator bugs that
  put gold span boundaries inside a token are fixed.

## 0.1.0

### Minor Changes

- [#6](https://github.com/AnandChowdhary/gpu-utils/pull/6) [`fd6abf7`](https://github.com/AnandChowdhary/gpu-utils/commit/fd6abf73329c3694ea73b0a72832b4068c170bfd) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-log: a universal log line parser that extracts timestamps, levels,
  sources, threads, key=value pairs, messages and stack frames from any text log format.
  JSON lines and pure logfmt are parsed deterministically; everything else goes through a
  150K-parameter int6 dilated-CNN token tagger that runs batched on WebGPU (CPU fallback).
