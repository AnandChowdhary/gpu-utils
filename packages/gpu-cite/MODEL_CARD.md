# Model card: gpu-cite

## Task
Parse freeform citation and reference strings into structured bibliographic fields

## Architecture
- Tokenizer: character-class runs; sparse hashed features per token (no learned vocabulary)
- Embedding: TODO dims
- Sequence mixing: TODO (bidirectional gated affine scans | dilated 1-D convolutions)
- Head: TODO labels; CPU Viterbi/CRF decode
- Parameters: TODO
- Quantization: int6 symmetric per-tensor, quantization-aware training

## Training data
TODO: sources, licenses, sizes, synthetic generators, teacher.

## Evaluation
| Set | Size | Metric | Score |
|---|---|---|---|
| held-out | TODO | exact match | TODO |

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | TODO |
| Cold start (device + pipelines + upload) | TODO |
| Warm call, 1 KB input | TODO |

## Limitations and intended use
TODO. Not a substitute for validation; outputs are probabilistic.

## Checkpoint
- Promoted: TODO (date, seed, epoch)
- Training command: `pnpm train`
