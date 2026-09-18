---
"gpu-email": patch
---

Migrate to the shared conv model family (`ConvTagger` / `convTaggerForward` /
`conv_tagger.wgsl`), retrained. The package no longer ships its own model layers, training
loop, Viterbi, CPU forward pass or WGSL kernel; the line-kind and BIO contact heads are tag
columns of one family model and the decoder is unchanged.
