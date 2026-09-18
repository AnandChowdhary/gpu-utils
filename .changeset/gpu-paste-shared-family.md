---
"gpu-paste": patch
---

Migrate to the shared model family, retrained. The learned part of gpu-paste is now a
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
