---
"gpu-log": minor
---

gpu-log v2: moves onto the shared conv-family model (`ConvTagger`, canonical WGSL kernel, no
package-specific shader), adds a first-class `host` field, widens the training generator
(cluster/supercomputer RAS logs, Windows CBS and Event Viewer text, Proxifier, multi-word
sources, bracketed/parenthesised source and thread variants, quoted and nested-brace values,
key=value prose inside messages) and ships a fresh 148-line unfamiliar evaluation set next
to the frozen v1 set.
