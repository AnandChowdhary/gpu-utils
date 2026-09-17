# gpu-utils: agent guide

Tiny task-specific models, trained from scratch, running on WebGPU in the browser.
Each package is one task, one model, zero runtime dependencies, under a strict Brotli
size budget. This file is the contract every contributor and every coding agent follows.
Read it fully before touching a package.

## The recipe (do not deviate without a written reason in the package README)

1. **CPU pre-pass.** `tokenize()` from `@gpu-utils/runtime` splits text into runs by
   character class. Each token gets a handful of sparse hashed feature ids (word hash,
   consonant-skeleton hash, shape, first/last char, length bucket, task-specific flags).
   No learned vocabulary. The featurizer in `src/features.ts` and
   `training/<snake>/features.py` must be byte-for-byte equivalent.
2. **Model.** Summed sparse embeddings (24–96 dims) → sequence mixing → small head.
   Two proven families:
   - *Affine-scan tagger* (gpu-lexer, gpu-time, gpu-query, gpu-cron): bidirectional gated
     affine scans `h = a*h_prev + b` as a parallel prefix scan, optional depthwise conv,
     mean-pooled global context, two-layer head. Best for short natural-language inputs
     (queries, schedules, commands). ~30–40K params.
   - *Dilated-CNN tagger* (gpu-pii, tinySarf): 3–5 residual dilated 1-D convs, kernel 3.
     Best for long documents (logs, prose, code). 100K–1M params.
   The model only tags tokens or emits a small set of roles. It never generates free text.
3. **Decoder.** Viterbi/CRF over the tags on the CPU, then a deterministic TypeScript
   compiler turns tags into the typed output (filter AST, cron string, spans, RRULE).
   All semantics, validation, and error messages live in the compiler, not the model.
4. **Training.** PyTorch, `uv`-managed, quantization-aware training to int6 (or int8 for
   models over 200K params). Data is synthetic-first, generated from a grammar with a
   teacher where one exists; real data is held out for evaluation. Training must run on
   CPU in under 30 minutes for the default config.
5. **Export.** `training/<snake>/export.py` writes `model/manifest.json`,
   `model/weights.txt` (int6 text encoding, see `tooling/python/gpu_utils_training/quant.py`)
   and `model/fixtures.json` (inputs, features, logits) used for parity tests.
6. **Runtime.** `src/cpu.ts` is the reference forward pass. `src/gpu.ts` plus
   `src/shader.wgsl` is the WebGPU forward pass and must match the CPU path to 1e-4.
   "auto" backend uses CPU for small inputs (GPU readback dominates below ~256 tokens).
   Never silently return empty results when WebGPU is missing: fall back to CPU.

## Package layout

```
packages/<name>/
  package.json          name, description, gpuUtils.sizeBudget (bytes, Brotli)
  README.md             install, one usage example, how it works, size table, limitations
  MODEL_CARD.md         architecture, params, data, eval tables, latency, checkpoint id
  src/index.ts          public API only: parse(text, options)
  src/features.ts       featurizer (parity with Python)
  src/cpu.ts            reference forward pass
  src/gpu.ts            WebGPU forward pass using createProgram()
  src/shader.wgsl       compute kernels
  src/decode.ts         Viterbi + tag→typed output compiler
  src/model.ts          loads model/manifest.json + weights.txt
  model/                promoted checkpoint artefacts (committed)
  test/                 vitest: features, decode, parity against model/fixtures.json
  training/             uv project: data.py, model.py, train.py, export.py, evaluate.py, tests/
video/scenes/<name>.md  storyboard
video/<snake>_pipeline.py  Manim explainer
```

Scaffold with `pnpm new <name> "<description>"`.

## Definition of done for a package

- `pnpm lint && pnpm typecheck && pnpm test && pnpm build && pnpm size && pnpm check:package` pass.
- `cd training && uv run pytest` passes; `uv run python -m <snake>.train` reproduces the
  promoted checkpoint's metrics within noise from the documented seed.
- Parity test: CPU logits match `model/fixtures.json` (from PyTorch) at 1e-4; WGSL matches CPU.
- Size is under budget. Default budget 40 KB Brotli for affine-scan models; set a
  justified budget in `package.json` for larger ones and say why in the README.
- MODEL_CARD.md has real numbers: params, size, held-out accuracy, a hard "unfamiliar"
  set the generator did not produce, latency cold/warm, and honest limitations.
- README has a working example, a "How it works" paragraph, and a limitations section.
- A storyboard and a rendered explainer (`bash video/render.sh <name>`).
- A changeset (`pnpm changeset`) describing the release.

## Conventions

- TypeScript strict, ESM only, no default exports except for `.wgsl`/`.txt` text imports.
- Public API is `parse(text, options)` returning a typed result, plus `parseMany` when
  batching matters. Offsets are UTF-16 code units, half-open.
- Errors are thrown `Error` subclasses with stable `name`s; never return partial results
  on failure.
- No runtime dependencies. `@gpu-utils/runtime` is bundled in at build time.
- Biome formats and lints; do not add ESLint or Prettier.
- Python: 3.12, `uv`, `ruff` defaults, type hints everywhere. Torch CPU wheels on Linux.
- Commit messages: Conventional Commits with the package as scope, e.g. `feat(gpu-view): add group-by roles`.
- Never commit training runs, checkpoints (`*.pt`), or rendered video. Commit exported
  `model/` artefacts and storyboards.

## Working as a subagent on one package

1. Read this file, `packages/runtime/src`, and the closest existing package.
2. Write the storyboard of the *data flow* first (what the tokens are, what the roles are,
   what the compiler emits). Put it in the README's "How it works" and the MODEL_CARD.
3. Build the featurizer in Python and TypeScript together, with shared fixtures.
4. Build the synthetic data generator and the evaluation sets before the model.
5. Train small, check the CPU path against fixtures, then write WGSL.
6. Only then measure size and latency and write the numbers down.
7. Report back with: metrics table, size, what the model cannot do, open questions.

Ask for a decision when: the task needs generation instead of tagging, the budget cannot
be met, or the eval shows the synthetic generator does not cover real inputs.

## References

- vercel-labs/gpu-lexer, arikchakma/gpu-time, safzanpirani/gpu-query,
  manuschillerdev/gpu-cron, f0rr0/gpu-postal, npm gpu-pii: the design lineage.
- `video/README.md` for the explainer rules.
