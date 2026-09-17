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

## Size and speed

TODO: table from `pnpm size` and the benchmark.

## Limitations

TODO. See [MODEL_CARD.md](./MODEL_CARD.md).

## Training

```bash
cd packages/__NAME__/training
uv sync
uv run python -m __SNAKE__.data      # build/generate dataset
uv run python -m __SNAKE__.train     # train with QAT
uv run python -m __SNAKE__.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
