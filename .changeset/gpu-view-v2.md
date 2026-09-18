---
"gpu-view": minor
---

gpu-view v2: moved onto the shared scan model family (canonical WGSL kernel, no package
kernel) and widened the synthetic generator's coverage categories. New in the featurizer and
compiler: a `primary` flag on schema fields for date disambiguation, comparative and
superlative adjectives resolved through field aliases (`cheapest first`, `taller than 50 cm`),
negation prefixes glued onto a boolean field word (`unarchived`, `nonbillable`, `inactive`),
unit nouns after a value (`longer than 60 minutes`), unit suffixes on numbers, full dates and
seasons (`september 10 2026`, `last winter`), year-like numeric fields with polarity bounds
(`vintage 2019 or older`), `top N` stranded before its sort field, enum values that override a
carrier noun, and a spec-normalisation pass that merges constraints split across clause
boundaries. Adds two further hand-written unfamiliar evaluation sets and keeps reporting the
first, which is now labelled contaminated.
