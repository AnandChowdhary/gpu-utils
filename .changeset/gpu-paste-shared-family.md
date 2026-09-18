---
"gpu-paste": patch
---

Migrate to the shared model family, retrained. The learned part of gpu-paste is now a
plain `ScanTagger` from the shared runtime (BIO span head = per-token tags, kind head =
pooled output): the CPU path uses `scanTaggerForward`, the WebGPU path `runScanTagger` on
the canonical `scan_tagger.wgsl` kernel (windows of a long paste run as one batched
dispatch), and training/export/evaluation go through `gpu_utils_training`. The package's
own model layers, fake-quant, training loop, Viterbi tables, `shader.wgsl` and hand-written
forward pass are gone; rules, detectors, decoder policy, data generator and the evaluation
sets are unchanged. Retrained from scratch with the same budget; see MODEL_CARD.md for the
old-vs-new numbers.
