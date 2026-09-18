# __NAME__

__DESCRIPTION__

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install __NAME__
```

```ts
import { parse } from "__NAME__";

const result = await parse("...");
```

## How it works

TODO: tokenizer → sparse features → model → decoder, in three sentences.

The model is the shared *scan* family from `@gpu-utils/runtime` (summed sparse
embeddings → bidirectional gated affine scans → mean-pooled context → per-token head),
so the CPU forward pass and the WGSL kernels are the runtime's canonical ones; this
package only owns the featurizer, the training data and the decoder/compiler.

## Size and speed

TODO: table from `pnpm size` and the benchmark.

## Limitations

TODO. See [MODEL_CARD.md](./MODEL_CARD.md).

## Training

```bash
cd packages/__NAME__/training
uv sync
uv run python -m __SNAKE__.data      # build/generate dataset
uv run python -m __SNAKE__.train     # train with QAT → runs/default/{best.pt,history.json}
uv run python -m __SNAKE__.evaluate  # held-out metrics for MODEL_CARD.md
uv run python -m __SNAKE__.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
uv run pytest                        # features, fixtures, WGSL parity (skips without an adapter)
```
