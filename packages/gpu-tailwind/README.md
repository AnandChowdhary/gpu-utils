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
| `full width on small screens, half on large` | `max-sm:w-full lg:w-1/2` |
| `muted text, pill, subtle shadow` | `text-gray-500 rounded-full shadow-sm` |
| `padding 16px, no border, white text in dark mode` | `p-[16px] border-0 dark:text-white` |
| `on hover, blue background and white text` | `hover:bg-blue-500 hover:text-white` |
| `white text on a blue 700 background` | `text-white bg-blue-700` |
| `gradient from purple to pink` | `bg-linear-to-r from-purple-500 to-pink-500` |
| `translucent black backdrop` | `bg-black/50` |
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
2. **Tag (GPU or CPU).** A 45,394-parameter `ScanTagger` from `@gpu-utils/runtime` — the
   shared scan family: bidirectional gated affine scans (`h = a·h_prev + (1−a)·tanh(u)` as a
   masked prefix scan), a residual depthwise conv, a mean-pooled context vector and a
   two-layer head — labels every token `PROPERTY`, `VALUE`, `VARIANT`, `SEP`, `NEG` or `O`
   (BIO scheme, Viterbi-decoded) and emits a segment-boundary score from an extra tag
   column. Weights are int6. The package ships no shader of its own: the WebGPU path runs
   the canonical `scan_tagger.wgsl` kernel from the runtime.
3. **Compile (CPU, deterministic).** `src/compile.ts` splits the phrase into segments at
   separators and boundary scores, resolves each `VARIANT` span with keyword rules
   (`on hover` → `hover:`, `on mobile` → `max-sm:`, `when the parent is hovered` →
   `group-hover:`), pairs every `VALUE` with the nearest compatible `PROPERTY` (`gap 4`,
   `blue background`, `bold red text`), and maps each pair through a table compiled from the
   Tailwind v4 default theme (`src/table.json`: 26 hues × 11 shades × 11 alphas, spacing,
   sizes, radii, shadows, breakpoints, and ~2,200 natural-language synonyms such as
   `subtle shadow` → `shadow-sm`, `pill` → `rounded-full`, `muted` → `text-gray-500`). A
   variant phrase that *leads* its segment ("on hover, blue and underlined") scopes over the
   following segments; a trailing one ("blue on hover") applies to its own. Values without a
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
| Package (min + Brotli, weights and table included) | 54.9 KiB / 56,254 B (budget 60,000 B) |
| Parameters | 45,394 (int6) |
| CPU path, 15-token phrase (Node 24, one core) | 0.48 ms |
| WebGPU cold start (device + pipelines + weight upload) | ~40-80 ms |
| WebGPU warm call, 300-token input | ~1-2 ms compute + ~1-3 ms readback |

The budget is above the 40 KB default because the compiled theme + synonym table
(`src/table.json`, 13.4 KiB Brotli) ships alongside the weights (25.3 KiB Brotli).

## Limitations

- Coverage is the compiled table: layout/display, flex/grid, spacing, sizing, typography,
  colours with shades and alpha, linear gradients, borders/radius/ring/outline, shadows,
  opacity, position/inset, overflow, z-index, transitions/animation,
  cursor/select/pointer-events, object-fit, aspect ratio, scale/rotate/blur, and
  responsive/state/pseudo variants. Transforms beyond scale/rotate, filters other than
  blur/grayscale, `backdrop-*`, custom theme tokens, arbitrary selectors and `!important`
  are out of scope.
- Defaults are opinionated and documented in the model card (`rounded` → `rounded-lg`,
  `on small screens` → `max-sm:`, a bare colour → `bg-*`). Phrases that need context the
  model cannot see ("same colour as the header") produce diagnostics.
- Some words are genuinely ambiguous and the compiler picks one reading: "bold red text"
  compiles to `text-red-600` (a bold *shade* of red), not `font-bold text-red-500`, because
  `bold` is both a font weight and a shade word.
- Three or more breakpoint clauses in one sentence ("small on mobile, base on tablet, large
  on desktop") lose the later ones. Long inputs with many segments accumulate tagging
  errors; the metrics are for phrases of 1–5 segments. See
  [MODEL_CARD.md](./MODEL_CARD.md) for the honest numbers, including a hand-written
  unfamiliar set the generator never produced (52.9% exact-set match, 80.6% class F1).

## Training

```bash
cd packages/gpu-tailwind/training
uv sync
uv run python -m gpu_tailwind.build_table   # theme.css + lexicon -> ../src/table.json
uv run python -m gpu_tailwind.data          # 130K synthetic + 6K held-out + oracle sample
uv run python -m gpu_tailwind.train         # QAT, ~17 min on 2 CPU threads
uv run python -m gpu_tailwind.export        # ../model/{manifest.json,weights.txt,fixtures.json}
uv run python -m gpu_tailwind.evaluate      # tag-level held-out metrics
uv run pytest                               # featurizer parity + WGSL vs fixtures (wgpu/lavapipe)
cd .. && pnpm eval                          # class-level exact-set match and per-class P/R
```
