---
"gpu-log": minor
---

Initial release of gpu-log: a universal log line parser that extracts timestamps, levels,
sources, threads, key=value pairs, messages and stack frames from any text log format.
JSON lines and pure logfmt are parsed deterministically; everything else goes through a
150K-parameter int6 dilated-CNN token tagger that runs batched on WebGPU (CPU fallback).
