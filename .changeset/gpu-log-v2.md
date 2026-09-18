---
"gpu-log": minor
---

gpu-log v2: the model moves onto the shared conv family (`ConvTagger`, the canonical
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
