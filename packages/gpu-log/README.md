# gpu-log

Universal log line parser: timestamps, levels, sources, key=value pairs, messages, and stack frames from any format

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-log
```

```ts
import { parse } from "gpu-log";

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
cd packages/gpu-log/training
uv sync
uv run python -m gpu_log.data      # build/generate dataset
uv run python -m gpu_log.train     # train with QAT
uv run python -m gpu_log.export    # write ../model/{manifest.json,weights.txt,fixtures.json}
```
