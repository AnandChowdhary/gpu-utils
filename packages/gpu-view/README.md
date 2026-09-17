# gpu-view

Natural language to table view specs: filter, sort, group, aggregate, limit, chart

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-view
```

```ts
import { parse } from "gpu-view";

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
cd packages/gpu-view/training
uv sync
uv run python -m gpu_view.data      # build/generate dataset
uv run python -m gpu_view.train     # train with QAT
uv run python -m gpu_view.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
