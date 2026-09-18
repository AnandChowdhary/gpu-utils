# Model card: __NAME__

## Task
__DESCRIPTION__

## Architecture
- Tokenizer: character-class runs; sparse hashed features per token (no learned vocabulary)
- Family: `ScanTagger` from `gpu_utils_training.models` (bidirectional gated affine scans,
  `h = a * h_prev + (1 - a) * tanh(u)`, residual 5-tap depthwise conv, mean-pooled context)
  — or `ConvTagger` (residual dilated 1-D convolutions) for long documents
- Embedding: TODO dims (`hidden`), TODO scan layers
- Head: TODO labels; CPU Viterbi with BIO constraints
- Parameters: TODO
- Quantization: int6 symmetric per-tensor, quantization-aware training from epoch 1

## Training data
TODO: sources, licenses, sizes, synthetic generators, teacher.

## Evaluation
| Set | Size | Metric | Score |
|---|---|---|---|
| held-out (generated) | TODO | span F1 / exact match | TODO |
| unfamiliar (hand-written) | TODO (≥ 60) | span F1 / exact match | TODO |

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | TODO |
| Cold start (device + pipelines + upload) | TODO |
| Warm call, 1 KB input | TODO |

## Limitations and intended use
TODO. Not a substitute for validation; outputs are probabilistic.

## Checkpoint
- Promoted: TODO (date, seed, epoch; see `manifest.json` → `checkpoint`)
- Training command: `pnpm train`
