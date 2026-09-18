---
"gpu-email": patch
---

Migrate to the shared conv model family (`ConvTagger` / `convTaggerForward` /
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
