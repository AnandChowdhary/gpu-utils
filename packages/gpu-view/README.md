# gpu-view

Natural language to table view specs: filter, sort, group, aggregate, limit, chart.

Tiny model (__PARAMS__ parameters, int6), trained from scratch, runs on WebGPU or the CPU in the
browser. Zero dependencies, 29.6 KiB Brotli including the weights. Part of
[gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

It is **schema-blind**: the model never sees your field names. You pass the schema at
call time and it is injected as anonymous per-token features ("this token matched a
field of kind `number`"), so the same weights work on issues, wine cellars and
greenhouses without retraining.

```bash
npm install gpu-view
```

```ts
import { parse } from "gpu-view";

const schema = {
  fields: [
    { name: "revenue", kind: "number", aliases: ["sales"] },
    { name: "region", kind: "enum", values: ["EMEA", "APAC", "Americas"] },
    { name: "closed_at", kind: "date", aliases: ["closed", "close date"] },
    { name: "owner", kind: "text", aliases: ["rep"] },
  ],
};

const spec = await parse("total revenue by region this quarter, top 10, as a bar chart", { schema });
// {
//   filters:   [{ field: "closed_at", op: "between", value: ["2026-07-01", "2026-09-30"], span: { start: 26, end: 38 } }],
//   sort:      [],
//   groupBy:   [{ field: "region", span: { start: 17, end: 23 } }],
//   aggregate: [{ fn: "sum", field: "revenue", span: { start: 0, end: 13 } }],
//   limit:     10,
//   chart:     "bar",
//   diagnostics: [],
//   tokens: [{ text: "total", start: 0, end: 5, role: "AGG_FN", boundary: true }, ...],
// }
```

Other phrases it handles: `open issues assigned to me sorted by priority`,
`customers in Germany or France with more than 5 orders`,
`average order value per month last year, line chart`, `repos not updated since 2024`,
`plants without a species, newest first, limit 25`.

### Options

| Option | Purpose |
|---|---|
| `schema` | `{ fields: [{ name, kind: "text" \| "number" \| "date" \| "enum" \| "boolean", aliases?, values? }] }`. Required. |
| `now` | Reference date for relative phrases (`Date` or ISO string). Defaults to the current time. |
| `dateField` | Date field to use when a time phrase names none and the schema has several date fields. |
| `backend` | `"auto"` (default), `"cpu"` or `"webgpu"`. A single phrase always runs on the CPU under `auto`; `parseMany` batches onto the GPU. |

### Output

`filters[].op` is one of `eq neq contains gt gte lt lte between in is_true is_false is_empty not_empty`.
Values are canonical enum strings, numbers (`5k` → `5000`, `$1,200` → `1200`), ISO dates or
`[from, to]` pairs. `sort[].dir` is `asc`/`desc`; `aggregate[].fn` is `count sum avg min max`;
`chart` is `table bar line pie number`; `granularity` is set for "per month"-style groups.
Every clause carries a `span` (UTF-16 offsets, half-open) into the input, and every token is
returned with its predicted role.

Anything the compiler cannot resolve becomes a `diagnostics` entry with a stable `code`
(`unknown_field`, `unknown_value`, `ambiguous_value`, `unresolved_time`,
`ambiguous_date_field`, `unsupported_negation`, ...) and a span, instead of a guess.

## How it works

`tokenize()` splits the phrase into character-class runs and whitespace is dropped. Each
token gets sparse feature ids: word and consonant-skeleton hashes, shape, length, a closed
lexicon of operator/sort/group/aggregate/chart/time words, and the schema-membership
features that make the model schema-blind: matched a field (its kind, begin/inside, exact
/ stem / prefix / typo, via an alias), matched an enum value (unique owner? owned by the
nearest preceding or following field?), and the kind of and distance to the nearest field
match on either side. The field words themselves never reach the model.

A __PARAMS__-parameter tagger from the shared scan family (`@gpu-utils/runtime`
`scanTaggerForward`; summed embeddings → bidirectional gated affine scan
`h[t] = a[t]·h[t−1] + (1−a[t])·tanh(u[t])` as a parallel prefix scan → residual 5-tap
depthwise convolution → mean-pooled context → two-layer head) emits one of 14 roles per
token (`FIELD OP VALUE TIME_VALUE CONJ NEG SORT_FIELD SORT_DIR GROUP_FIELD AGG_FN AGG_FIELD
LIMIT CHART O`) plus a clause-boundary logit, so several filters in one phrase stay apart.

Deterministic TypeScript then compiles the roles: it resolves field spans back to your
schema (aliases, plurals, prefixes, one-character typos), canonicalises enum values, parses
numbers with `k`/`m` suffixes and currency symbols, resolves relative dates (`today`,
`last 30 days`, `this quarter`, `q2 2025`, `march`, a bare year, ISO dates) against `now`,
maps operator phrases and negation onto the op set, pairs aggregate functions with their
fields, and reports diagnostics for anything left over. Comparatives and superlatives
resolve through aliases: list `cheap` or `tall` as an alias of a numeric field and
`cheapest first` sorts ascending, `taller than 50 cm` becomes `gt 50`. Mark one date field
`primary: true` so bare time phrases ("this quarter") attach to it when the schema has
several.

## Size and speed

| Measure | Value |
|---|---|
| Bundle (min + Brotli, weights included) | 29.6 KiB (budget 39.1 KiB) |
| Parameters | 33,087 (int6) |
| Import + weight decode | ~5 ms (Node 24, Xeon 2.9 GHz) |
| First parse (JIT warm-up) | ~15 ms |
| Warm CPU parse, 19-token phrase | ~0.85 ms |
| CPU batch of 1000 phrases | ~640 ms |

The WGSL kernel (one workgroup per phrase, one thread per hidden channel) matches the CPU
path to 1e-5 in a batched dispatch on Mesa's software Vulkan; it is used by `parseMany` for
batches, where readback cost is amortised. Real-GPU timings are not measured yet.

## Limitations

- Trained entirely on synthetic phrases. Held-out generated schemas reach 91% exact spec
  match; a hand-written set of 65 phrases over four unseen schemas reaches 55%. See
  [MODEL_CARD.md](./MODEL_CARD.md) for the failure analysis.
- Superlatives that are not in the lexicon (`cheapest`, `tallest`), inflections the matcher
  cannot bridge (`unopened` vs `opened`, `rated` vs `rating`), units after numbers
  (`50 cm`) and date formats outside the resolver (`september 10 2026`, `this spring`)
  are reported as diagnostics or missed.
- Filters are always ANDed; `or` only forms value lists. Negated `contains`/`between` and
  negated date ranges are unsupported and reported.
- `top N` without a `by` field only sets `limit`; it does not infer a sort from an aggregate.
- "Per month" needs exactly one date field in the schema or `options.dateField`.
- English only.

## Training

```bash
cd packages/gpu-view/training
uv sync
uv run python -m gpu_view.data      # fixtures, eval sets, 160K-phrase corpus (~4 min)
uv run python -m gpu_view.train     # 14 epochs of int6 QAT on 2 CPU threads (~12 min)
uv run python -m gpu_view.export    # ../model/{manifest.json,weights.txt,fixtures.json}
uv run pytest                       # featurizer/time/matcher parity, generator invariants, WGSL via wgpu-py
```
