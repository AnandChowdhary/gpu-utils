# gpu-utils

Tiny libraries for doing things in the browser with WebGPU.

Each package is a small model, trained from scratch, that runs on the user's GPU in the
browser: no server, no API key, no network round trip, and a bundle measured in tens of
kilobytes. The approach follows [gpu-lexer](https://github.com/vercel-labs/gpu-lexer) and
[gpu-time](https://github.com/arikchakma/gpu-time): a mechanical tokenizer, hashed sparse
features, a ~30K-parameter tagger in hand-written WGSL, and a deterministic compiler that
turns tags into typed output.

## Packages

| Package | Task | Size | Status |
|---|---|---|---|
| _coming soon_ | | | |

## Development

```bash
pnpm install
pnpm lint && pnpm typecheck && pnpm test
pnpm build && pnpm size && pnpm check:package

pnpm new gpu-thing "Natural language to thing"   # scaffold a package
cd packages/gpu-thing/training && uv sync && uv run python -m gpu_thing.train
bash video/render.sh gpu-thing draft             # explainer video
```

Node 24, pnpm 11, Python 3.12 via [uv](https://docs.astral.sh/uv/). See [AGENTS.md](./AGENTS.md)
for the full recipe and definition of done.

## Releasing

Add a changeset (`pnpm changeset`). Merging to `main` opens a "Version Packages" pull
request; merging that publishes to npm with provenance. Requires the `NPM_TOKEN` secret.

## License

MIT
