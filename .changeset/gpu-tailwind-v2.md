---
"gpu-tailwind": minor
---

gpu-tailwind v2: move the model onto the shared `ScanTagger` family and the canonical
`scan_tagger.wgsl` kernel from `@gpu-utils/runtime` (the package no longer ships its own
shader), widen the synthetic generator to cover gradients, colour alpha, shade wording,
two-colour "X on a Y background" units, number words and heavier casing/typo noise, and
teach the compiler leading-variant scoping. Adds a second hand-written unfamiliar
evaluation set and makes the held-out split disjoint from the training set.
