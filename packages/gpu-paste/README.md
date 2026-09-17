# gpu-paste

Understand pasted text: detect what it is and extract structured fields, on-device

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-paste
```

```ts
import { parse } from "gpu-paste";

const result = await parse("...");
```

## How it works

TODO: tokenizer → sparse features → model → decoder, in three sentences.

## Size and speed

TODO: table from `pnpm size` and the benchmark.

## Limitations

TODO. See [MODEL_CARD.md](./MODEL_CARD.md).

## Training

```bash
cd packages/gpu-paste/training
uv sync
uv run python -m gpu_paste.data      # build/generate dataset
uv run python -m gpu_paste.train     # train with QAT
uv run python -m gpu_paste.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
