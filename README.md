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
| [gpu-cite](./packages/gpu-cite) | Parse citation and reference strings into structured bibliographic fields | 54.1 KiB | beta |
| [gpu-email](./packages/gpu-email) | Split plain-text emails into reply, quoted history, and signature, and extract contact details | 92.9 KiB | experimental |
| [gpu-paste](./packages/gpu-paste) | Understand pasted text: detect what it is and extract structured fields | 51.0 KiB | experimental |
| [gpu-view](./packages/gpu-view) | Natural language to table view specs: filter, sort, group, aggregate, limit, chart | 29.6 KiB | experimental |

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
request; merging that publishes to npm with provenance via npm Trusted Publishing (OIDC), so no
npm token is stored in the repo.

## License

MIT
