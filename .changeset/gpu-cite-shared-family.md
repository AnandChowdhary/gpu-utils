---
"gpu-cite": patch
---

Migrate to the shared `ScanTagger` model family from `@gpu-utils/runtime` / `gpu_utils_training`, retrained. The name-part head now lives in three extra tag columns, the document type in the pooled head, and the CRF transition matrix is an extra exported tensor used only by the decoder; the CPU path uses the runtime's `scanTaggerForward`, the WebGPU path the canonical `scan_tagger.wgsl` kernel, and the package no longer ships its own shader or forward pass.
