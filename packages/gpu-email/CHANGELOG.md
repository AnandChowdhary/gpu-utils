# gpu-email

## 0.1.0

### Minor Changes

- [#5](https://github.com/AnandChowdhary/gpu-utils/pull/5) [`bb38b20`](https://github.com/AnandChowdhary/gpu-utils/commit/bb38b20198df7268dcedf29ed831f0e335f7288a) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-email: splits plain-text emails into reply, attribution, quote, signature, disclaimer, forward-header, greeting and closing segments, extracts the new reply text, and pulls name, title, company, phone, email, URL and address out of the author's signature. Dilated-CNN two-head tagger (~155K params, int6) with exact rules for quote prefixes, the "-- " delimiter and email/URL regexes; WebGPU forward pass with a CPU fallback.
