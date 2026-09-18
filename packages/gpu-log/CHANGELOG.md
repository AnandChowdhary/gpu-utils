# gpu-log

## 0.1.0

### Minor Changes

- [#6](https://github.com/AnandChowdhary/gpu-utils/pull/6) [`fd6abf7`](https://github.com/AnandChowdhary/gpu-utils/commit/fd6abf73329c3694ea73b0a72832b4068c170bfd) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-log: a universal log line parser that extracts timestamps, levels,
  sources, threads, key=value pairs, messages and stack frames from any text log format.
  JSON lines and pure logfmt are parsed deterministically; everything else goes through a
  150K-parameter int6 dilated-CNN token tagger that runs batched on WebGPU (CPU fallback).
