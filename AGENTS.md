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
   `training/<snake>/features.py` must be byte-for-byte equivalent. Give every feature
   family its own block of ids in one embedding table (see the template) so ids never
   collide; `slots` is the fixed number of ids per token.
2. **Model: use the shared families.** `gpu_utils_training.models` provides two reference
   families with one interface, `forward(rows, mask) -> {"tags": [B,T,L], "pooled": [B,P] | None}`,
   whose CPU forward (`packages/runtime/src/models.ts`) and WGSL kernels
   (`packages/runtime/src/wgsl/{scan,conv}_tagger.wgsl`) already exist and are parity-tested
   on every CI run:
   - *Scan family* `ScanTagger(feature_rows, hidden, tags, pooled_out=0, scan_layers=1)`:
     summed sparse embeddings → bidirectional gated affine scans
     `h = a*h_prev + (1-a)*tanh(u)`, `a = sigmoid(...)` → residual 5-tap depthwise conv →
     mean-pooled context → two-layer head (+ optional pooled head). Best for short
     natural-language inputs (queries, schedules, commands, references). ~30–60K params.
   - *Conv family* `ConvTagger(feature_rows, embed, hidden, blocks, dilations, tags, pooled_out=0)`:
     embeddings → projection → residual blocks `x + relu(conv3_dilated(x)) @ W2` →
     per-token head (+ optional pooled head). Best for long documents (logs, prose, email).
     100K–1M params.
   **Only write a custom `nn.Module` (with matching cpu.ts and shader.wgsl) if the package
   README explains why neither family fits.** Extra heads (a boundary logit, a kind
   classifier) fit the families: use `tags` for per-token outputs and `pooled_out` for
   per-sequence outputs, then split the logits in the decoder.
   The model only tags tokens or emits a small set of roles. It never generates free text.
3. **Decoder.** Viterbi over the tags on the CPU (`viterbi` + `bioTransitions` +
   `bioToSpans` from the runtime), then a deterministic TypeScript compiler turns tags into
   the typed output (filter AST, cron string, spans, RRULE). All semantics, validation, and
   error messages live in the compiler, not the model.
4. **Training.** PyTorch CPU, `uv`-managed, through `gpu_utils_training.loop.train`
   (AdamW, warm-up + cosine, int6 QAT from epoch 1, evaluation of the quantized model every
   epoch, best checkpoint, `history.json`, wall-clock budget). Data is synthetic-first,
   generated from a grammar with a teacher where one exists; real data is held out for
   evaluation. Training must run on CPU in under 30 minutes for the default config.
5. **Export.** `gpu_utils_training.export.export_package` writes `model/manifest.json`,
   `model/weights.txt` (int6 text encoding, see `quant.py`) and `model/fixtures.json` in
   the canonical format `{"cases": [{"input", "rows", "logits", "pooled"}]}`, computed from
   the decoded int6 weights so parity is exact.
6. **Runtime.** `src/cpu.ts` calls the family forward from the runtime (the reference).
   `src/gpu.ts` calls `runScanTagger` / `runConvTagger` on the canonical kernels and must
   match the CPU path to 1e-4 (`training/tests/test_wgsl.py` checks this on Mesa lavapipe
   in CI). "auto" backend uses CPU for small inputs (GPU readback dominates below ~256
   tokens). Never silently return empty results when WebGPU is missing: fall back to CPU.

## Package layout

```
packages/<name>/
  package.json          name, description, gpuUtils.sizeBudget (bytes, Brotli)
  README.md             install, one usage example, how it works, size table, limitations
  MODEL_CARD.md         architecture, params, data, eval tables, latency, checkpoint id
  src/index.ts          public API only: parse(text, options)
  src/features.ts       featurizer (parity with Python)
  src/cpu.ts            reference forward pass (family forward from the runtime)
  src/gpu.ts            WebGPU forward pass (runScanTagger / runConvTagger)
  src/decode.ts         Viterbi + tag→typed output compiler
  src/model.ts          loads model/manifest.json + weights.txt
  src/shader.wgsl       ONLY for a justified custom kernel (see step 2)
  model/                promoted checkpoint artefacts (committed)
  test/                 vitest: features, decode, parity against model/fixtures.json
  training/             uv project: data.py, model.py, train.py, evaluate.py, export.py, tests/
video/scenes/<name>.md  storyboard
video/<snake>_pipeline.py  Manim explainer
```

Scaffold with `pnpm new <name> "<description>"`. The scaffold ships a random-weight
scan model and a placeholder task so every check passes before you train.

## Shared modules

`tooling/python/gpu_utils_training` (installed editable into every package; depends on
torch CPU, numpy, wgpu so packages do not redeclare them):

| Module | What it gives you |
|---|---|
| `features` | `tokenize`, `hash_token`: byte-for-byte mirror of the runtime tokenizer |
| `layers` | `sparse_embed`, `affine_scan` (masked Hillis-Steele, identity on padding), `BiScan`, `DepthwiseConv`, `DilatedResidualBlock`, `Dense`, `masked_mean` |
| `models` | `ScanTagger`, `ConvTagger`, `from_config`; `.tensors()` in runtime layout, `.config()`, `.load_tensors()` |
| `qat` | `fake_quant` (STE, bit-identical to the runtime decoder), `QParam`, `set_quant`, `QuantMixin` |
| `quant` | int6 `quantize`/`encode`/`export`, `decode_weights`/`flat_weights` (what `weights.ts` computes) |
| `batch` | `pack_rows`, `collate`, `pad_labels`: the `[B, T, slots]` layout shared with `batch.ts` |
| `loop` | `train(model, make_batches, evaluate, loss=..., epochs, lr, weight_decay, qat_from, threads, seed, out_dir, select, minutes)`, `load_checkpoint` |
| `metrics` | `bio_to_spans`, `span_prf` (micro/macro/per label), `exact_match`, `token_accuracy`, `confusion`, `format_prf_table` |
| `decode` | NumPy `viterbi`, `bio_transitions`, `bio_start_mask` (fixture-tested against `decode.ts`) |
| `export` | `export_package(model, out_dir, labels, extra, fixtures)`, `check_fixtures` |
| `cli` | `training_parser` (`--epochs --seed --threads --minutes --run --lr --batch ...`), `run_dir`, `loop_kwargs` |
| `wgsl` | `WgslRunner(shader, entries).run(buffers, passes, readback)` on wgpu-py, mirroring `program.ts` |
| `kernels` | `make_runner`, `run_tagger`: run the canonical kernels from Python exactly like `gpu.ts` |
| `testing` | pytest fixture `wgpu_device` (skips cleanly without an adapter) |
| `fixtures` | regenerates `packages/runtime/test/fixtures/*.json` (`uv run python -m gpu_utils_training.fixtures`) |

`packages/runtime/src` (bundled into every package, tree-shakeable — unused families do
not end up in a package bundle):

| Module | What it gives you |
|---|---|
| `tokenize`, `hash` | tokenizer and FNV-1a hashing |
| `layers` | `sparseEmbed`, `dense`, `affineScan`, `biScan`, `depthwiseConv`, `dilatedResidualBlock`, `meanPool`, `concatRows`, `relu`, `sigmoid` |
| `models` | `scanTaggerForward`, `convTaggerForward`, `taggerForward`, `scanTensorNames`, `convTensorNames`, `TaggerManifest` |
| `decode`, `bio` | `viterbi`, `argmax`, `bioTransitions`, `bioStartMask`, `bioToSpans` |
| `batch` | `packRows`, `tensorOffsets`, `taggerParams`, `grid`, `unpackLogits` |
| `gpu` | `runScanTagger`, `runConvTagger`, `scanTaggerEntries`, `convTaggerEntries`, `scanTaggerShader`, `convTaggerShader` |
| `program`, `device`, `weights` | `createProgram`, `getDevice`, `decodeInt6`, `tensor` |

Kernel conventions (for custom kernels too): binding index = position in the buffer list;
every entry point must statically reference every binding (the runtime builds one bind
group per pipeline); dense weights are `[in, out]`; no invocation may loop more than a few
thousand times (Mesa lavapipe, the CI adapter, aborts an invocation after ~65K loop
iterations, so split work into per-token passes and keep only the scan recurrence
sequential).

## Definition of done for a package

- `pnpm lint && pnpm typecheck && pnpm test && pnpm build && pnpm size && pnpm check:package` pass.
- `cd training && uv run pytest` passes (features, fixtures, WGSL parity on lavapipe);
  `uv run python -m <snake>.train` reproduces the promoted checkpoint's metrics within
  noise from the documented seed.
- Parity test: CPU logits match `model/fixtures.json` (from PyTorch) at 1e-4; WGSL matches CPU.
- Size is under budget. Default budget 40 KB Brotli for scan-family models; set a
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
- Python: 3.12, `uv`, `ruff` defaults, type hints everywhere. Torch CPU wheels on Linux
  (resolved through `gpu-utils-training`; keep the `pytorch-cpu` index in `pyproject.toml`).
- Commit messages: Conventional Commits with the package as scope, e.g. `feat(gpu-view): add group-by roles`.
- Never commit training runs, checkpoints (`*.pt`), or rendered video. Commit exported
  `model/` artefacts and storyboards.

## Working as a subagent on one package

1. Read this file, `packages/runtime/src`, `tooling/python/gpu_utils_training`, and the
   closest existing package.
2. Write the storyboard of the *data flow* first (what the tokens are, what the roles are,
   what the compiler emits). Put it in the README's "How it works" and the MODEL_CARD.
3. Build the featurizer in Python and TypeScript together, with shared fixtures.
4. Build the synthetic data generator and the evaluation sets before the model.
5. Pick a family, train small, export, check `pnpm test` (CPU parity) and
   `uv run pytest` (WGSL parity). Only then measure size and latency and write the
   numbers down.
6. Report back with: metrics table, size, what the model cannot do, open questions.

Ask for a decision when: the task needs generation instead of tagging, the budget cannot
be met, the eval shows the synthetic generator does not cover real inputs, or you believe
a custom model is needed.

## References

- vercel-labs/gpu-lexer, arikchakma/gpu-time, safzanpirani/gpu-query,
  manuschillerdev/gpu-cron, f0rr0/gpu-postal, npm gpu-pii: the design lineage.
- `video/README.md` for the explainer rules.
