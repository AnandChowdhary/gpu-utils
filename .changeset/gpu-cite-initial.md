---
"gpu-cite": minor
---

Add `gpu-cite`: parse freeform citation and reference strings (APA, MLA, Chicago, IEEE, Vancouver, Harvard, Nature, ACM, arXiv listings, messy copy-paste) into structured bibliographic fields. A 76K-parameter int6 bidirectional affine-scan BIO tagger with a name-part head and a document-type head, deterministic DOI/arXiv/URL extraction, `parse()` and `parseMany()` with UTF-16 spans, CPU reference path and WebGPU kernels.
