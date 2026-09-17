# gpu-tailwind

Natural language to Tailwind CSS utility classes

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-tailwind
```

```ts
import { parse } from "gpu-tailwind";

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
cd packages/gpu-tailwind/training
uv sync
uv run python -m gpu_tailwind.data      # build/generate dataset
uv run python -m gpu_tailwind.train     # train with QAT
uv run python -m gpu_tailwind.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
