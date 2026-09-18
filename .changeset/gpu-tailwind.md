---
"gpu-tailwind": minor
---

Add gpu-tailwind: natural language to Tailwind CSS v4 utility classes. A ~45K-parameter
bidirectional affine-scan tagger labels tokens as property / value / variant / separator /
negation and scores segment boundaries; a deterministic compiler maps (property, value)
pairs to classes through a table compiled from the Tailwind v4 default theme, applies
responsive and state variants per phrase segment, and validates every class against the
compiled vocabulary. Runs on WebGPU with a CPU reference path.
