# gpu-tailwind

Natural language to Tailwind CSS utility classes.

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-tailwind
```

```ts
import { parse } from "gpu-tailwind";

const result = await parse(
  "card with rounded corners, subtle shadow, blue on hover, hidden on mobile",
);

result.classes;
// ["rounded-lg", "bg-white", "p-4", "shadow-sm", "hover:bg-blue-500", "max-sm:hidden"]

result.groups;
// [
//   { span: { start: 0, end: 4 },   text: "card",            classes: ["rounded-lg", "bg-white", "p-4", "shadow-sm"] },
//   { span: { start: 10, end: 25 }, text: "rounded corners", classes: ["rounded-lg"] },
//   { span: { start: 27, end: 40 }, text: "subtle shadow",   classes: ["shadow-sm"] },
//   { span: { start: 42, end: 55 }, text: "blue on hover",   classes: ["hover:bg-blue-500"], variant: "hover" },
//   { span: { start: 57, end: 73 }, text: "hidden on mobile", classes: ["max-sm:hidden"],   variant: "max-sm" },
// ]

result.diagnostics;
// [] — unknown words, values without a property, or variants with nothing to apply to
//       are reported here instead of being guessed
```

More examples:

| Phrase | Classes |
|---|---|
| `centered flex row with gap 4` | `flex items-center justify-center flex-row gap-4` |
| `bold red text, small, uppercase` | `font-bold text-red-500 text-sm uppercase` |
| `full width on small screens, half on large` | `max-sm:w-full lg:w-1/2` |
| `muted text, pill, subtle shadow` | `text-gray-500 rounded-full shadow-sm` |
| `padding 16px, no border, white text in dark mode` | `p-[16px] border-0 dark:text-white` |
| `hover:bg-blue-600 and p-4` (literal classes pass through) | `hover:bg-blue-600 p-4` |

Options: `parse(text, { backend: "auto" | "webgpu" | "cpu" })`. `"auto"` runs the CPU
reference path for short phrases (GPU readback dominates below ~256 tokens) and WebGPU
otherwise; environments without WebGPU always fall back to the CPU path. `parseMany(texts)`
batches calls.

## How it works

1. **Tokenize and featurize (CPU).** The shared gpu-utils tokenizer splits the phrase into
   character-class runs. Each token gets 7 hashed sparse features (word, consonant skeleton,
   3-char prefix and suffix, shape, length, class) into one 1,565-row embedding table; there is
   no learned vocabulary, so typos and British/American spellings still land near their
   neighbours.
2. **Tag (GPU or CPU).** A {{PARAMS}}-parameter bidirectional gated affine-scan tagger
   (`h = a·h_prev + b` as a parallel prefix scan, a depthwise conv, mean-pooled context and a
   two-layer head) labels every token `PROPERTY`, `VALUE`, `VARIANT`, `SEP`, `NEG` or `O`
   (BIO scheme, Viterbi-decoded) and emits a segment-boundary score. Weights are int6.
3. **Compile (CPU, deterministic).** `src/compile.ts` splits the phrase into segments at
   separators and boundary scores, resolves each `VARIANT` span with keyword rules
   (`on hover` → `hover:`, `on mobile` → `max-sm:`, `when the parent is hovered` →
   `group-hover:`), pairs every `VALUE` with the nearest compatible `PROPERTY` (`gap 4`,
   `blue background`, `bold red text`), and maps each pair through a table compiled from the
   Tailwind v4 default theme (`src/table.json`: 26 hues × 11 shades, spacing, sizes, radii,
   shadows, breakpoints, and ~2,000 natural-language synonyms such as `subtle shadow` →
   `shadow-sm`, `pill` → `rounded-full`, `muted` → `text-gray-500`). Values without a
   property fall back to sensible defaults (a bare colour is a background, `bold` is a font
   weight, `centered` is `flex items-center justify-center`). Every emitted class is checked
   against the compiled vocabulary; anything that fails validation becomes a diagnostic,
   never an invalid class.

The model never generates text. Semantics, defaults and validation all live in the
compiler, which is mirrored line for line by `training/gpu_tailwind/semantics.py` so the
synthetic generator's gold labels and the runtime agree (`test/oracle.test.ts`).

## Size and speed

| Measure | Value |
|---|---|
| Package (min + Brotli, weights and table included) | {{SIZE}} (budget 60 KB) |
| Parameters | {{PARAMS}} (int6) |
| CPU path, 15-token phrase (Node 24, one core) | {{CPU_LATENCY}} |
| WebGPU cold start (device + 6 pipelines + weight upload) | {{GPU_COLD}} |
| WebGPU warm call, 300-token input | {{GPU_WARM}} |

The budget is above the 40 KB default because the compiled theme + synonym table
(`src/table.json`, ~{{TABLE_SIZE}} Brotli) ships alongside the weights.

## Limitations

- Coverage is the compiled table: layout/display, flex/grid, spacing, sizing, typography,
  colours with shades, borders/radius/ring/outline, shadows, opacity, position/inset,
  overflow, z-index, transitions/animation, cursor/select/pointer-events, object-fit,
  aspect ratio, scale/rotate/blur, and responsive/state/pseudo variants. Gradients,
  transforms beyond scale/rotate, filters, custom theme tokens, arbitrary selectors and
  `!important` are out of scope.
- Defaults are opinionated and documented in the model card (`rounded` → `rounded-lg`,
  `on small screens` → `max-sm:`, a bare colour → `bg-*`). Phrases that need context the
  model cannot see ("same colour as the header") produce diagnostics.
- Long inputs with many segments accumulate tagging errors; the held-out metrics are for
  phrases of 1–5 segments. See [MODEL_CARD.md](./MODEL_CARD.md) for the honest numbers.

## Training

```bash
cd packages/gpu-tailwind/training
uv sync
uv run python -m gpu_tailwind.build_table   # theme.css + lexicon -> ../src/table.json
uv run python -m gpu_tailwind.data          # 130K synthetic + 6K held-out + oracle sample
uv run python -m gpu_tailwind.train         # QAT, ~14 min on 2 CPU threads
uv run python -m gpu_tailwind.export        # ../model/{manifest.json,weights.txt,fixtures.json}
uv run python -m gpu_tailwind.evaluate      # tag-level held-out metrics
uv run pytest                               # featurizer parity + WGSL vs fixtures (wgpu/lavapipe)
cd .. && pnpm eval                          # class-level exact-set match and per-class P/R
```
