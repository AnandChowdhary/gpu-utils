# gpu-cite

## 0.1.0

### Minor Changes

- [#2](https://github.com/AnandChowdhary/gpu-utils/pull/2) [`8f4caad`](https://github.com/AnandChowdhary/gpu-utils/commit/8f4caad9969064e811e16ecb7f9bbf980d201ba0) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Add `gpu-cite`: parse freeform citation and reference strings (APA, MLA, Chicago, IEEE, Vancouver, Harvard, Nature, ACM, arXiv listings, messy copy-paste) into structured bibliographic fields. A 76K-parameter int6 bidirectional affine-scan BIO tagger with a name-part head and a document-type head, deterministic DOI/arXiv/URL extraction, `parse()` and `parseMany()` with UTF-16 spans, CPU reference path and WebGPU kernels.
