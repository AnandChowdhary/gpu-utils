# gpu-email

## 0.1.1

### Patch Changes

- [#12](https://github.com/AnandChowdhary/gpu-utils/pull/12) [`a3b9a36`](https://github.com/AnandChowdhary/gpu-utils/commit/a3b9a36cedfdc1d6f01a498d46624dcffd259430) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Migrate to the shared conv model family (`ConvTagger` / `convTaggerForward` /
  `conv_tagger.wgsl`) and retrain. The package no longer ships its own model layers, training
  loop, Viterbi or WGSL kernel; the line-kind and BIO contact heads are tag columns of one
  family model, split in the decoder. The public API, the decoder rules and the evaluation
  sets are unchanged.
  
  Held-out generated set is a wash (reply exact match 0.9887 → 0.9870); the hand-written
  unfamiliar set improves (line-kind accuracy 0.8909 → 0.9074, reply exact match 0.7183 →
  0.7606) and the real email_reply_parser/talon fixtures improve (28/34 → 29/34 replies,
  2/6 → 3/6 signature blocks). The model grows from 155,207 to 170,519 parameters and the
  bundle from 93.0 KiB to 108.9 KiB Brotli (budget 117.2 KiB), and the CPU path is ~20%
  slower. See MODEL_CARD.md for both releases side by side.

## 0.1.0

### Minor Changes

- [#5](https://github.com/AnandChowdhary/gpu-utils/pull/5) [`bb38b20`](https://github.com/AnandChowdhary/gpu-utils/commit/bb38b20198df7268dcedf29ed831f0e335f7288a) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-email: splits plain-text emails into reply, attribution, quote, signature, disclaimer, forward-header, greeting and closing segments, extracts the new reply text, and pulls name, title, company, phone, email, URL and address out of the author's signature. Dilated-CNN two-head tagger (~155K params, int6) with exact rules for quote prefixes, the "-- " delimiter and email/URL regexes; WebGPU forward pass with a CPU fallback.
