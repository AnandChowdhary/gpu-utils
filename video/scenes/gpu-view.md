# How gpu-view turns a search phrase into a table view

## Overview

- **Topic**: A 33K-parameter tagger that reads "total revenue by region this quarter, top 10, as a bar chart" and emits a typed view spec, without ever seeing the app's field names.
- **Hook**: The model is schema-blind. Your schema enters as anonymous "matched a field of kind number" features, so the same weights work on issues, wine cellars and greenhouses.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 45–60 seconds, silent, 16:9.
- **Key Insight**: Tagging, not generating. The model labels tokens with roles and clause boundaries; deterministic TypeScript resolves fields, numbers and dates and emits diagnostics instead of guessing.

## Narrative Arc

Start with one phrase a user would type into a search bar. Split it into the tokens the model sees. Attach the schema as anonymous membership features next to each token (the field words are dropped at the boundary). Flow the same cells through the bidirectional gated affine scan. Reveal one role per token and the clause boundaries. Finish with the compiler assembling the view spec, with a diagnostic for an unresolved word, all in the browser.

---

## Scene 1: A search phrase

**Duration**: ~6 seconds

**Purpose**: Establish the input and the schema side by side.

### Visual Elements

- Caption: "A phrase in a search bar"
- The phrase `total revenue by region this quarter, top 10, as a bar chart` as `TokenCell`s (spaces muted).
- Footnote: `schema: revenue:number region:enum closed_at:date` in muted grey.

### Technical Notes

- Use `Text` with the bundled font, not `Code`.

---

## Scene 2: Tokens the model sees

**Duration**: ~6 seconds

**Purpose**: Show whitespace dropped and the remaining cells persisting.

### Visual Elements

- Caption: "Character-class tokens, whitespace dropped"
- Space cells fade out; word and punctuation cells slide together (`ReplacementTransform`).

---

## Scene 3: Schema becomes anonymous features

**Duration**: ~10 seconds

**Purpose**: The core idea. The field names never reach the model.

### Visual Elements

- Caption: "The schema enters as anonymous features"
- Under `revenue`, `region`, `quarter`: small muted tags `field:number`, `field:enum`, `time`. Under `total`, `top`, `bar`: `lexicon`.
- The literal names `revenue`, `region` in the schema footnote cross out; the tags stay.
- `activation_strip`s appear under every cell (hash, shape, length).
- Footnote: "no vocabulary, no field names, no server"

---

## Scene 4: Bidirectional gated scan

**Duration**: ~10 seconds

**Purpose**: Show context flowing both ways through the same cells.

### Visual Elements

- Caption: "h[t] = a[t]·h[t−1] + b[t], forward and backward"
- Accent-colored sweep left→right then right→left over the strips; strips brighten as context accumulates.
- Footnote: "33,087 parameters, int6"

---

## Scene 5: Roles and clause boundaries

**Duration**: ~8 seconds

**Purpose**: The model's only output: one role per token plus a boundary bit.

### Visual Elements

- Caption: "One role per token, plus clause boundaries"
- Role labels appear under cells in semantic colors: AGG_FN, AGG_FIELD, GROUP_FIELD, TIME_VALUE, SORT_DIR, LIMIT, CHART. `by`, `this`, `as`, `a` stay grey (O).
- Thin vertical ticks before `total`, `region`, `this`, `top`, `bar` mark boundaries.

---

## Scene 6: The compiler assembles the view

**Duration**: ~10 seconds

**Purpose**: Deterministic TypeScript resolves fields, dates and numbers.

### Visual Elements

- Caption: "Ordinary TypeScript compiles the roles"
- Role groups transform into spec lines:
  `aggregate: sum(revenue)` · `groupBy: region` · `filter: closed_at between 2026-07-01 … 2026-09-30` · `limit: 10` · `chart: bar`
- Footnote: "unresolved words become diagnostics, never guesses"

---

## Scene 7: Hold

**Duration**: ~3 seconds

**Purpose**: End on the assembled spec.

### Visual Elements

- Caption: "gpu-view · runs in your browser"
- Hold the spec.

---

## Recurring Visual Motifs

- Token cells (`TokenCell`) persist from decomposition to final labels.
- Accent `#3B82F6` marks learned state only.

## Color Palette

| Role | Hex | Usage |
|---|---|---|
| Background | #000000 | frame |
| Text | #F5F5F5 | primary |
| Muted | #999999 | footnotes, whitespace glyphs |
| Accent | #3B82F6 | activations, scan windows |
| Roles | #22C55E / #F59E0B / #A855F7 / #EC4899 / #14B8A6 | model outputs only |

## Accuracy Guardrails

- The animation must not imply the model sees field names: the schema footnote is crossed out before features appear, and the tags shown are the kinds (`field:number`), never the names.
- The animation must not imply the output is deterministic text generation: the model emits roles and boundaries; the spec is assembled by the compiler scene.
- Every number shown (33,087 parameters, int6) matches MODEL_CARD.md.
- Roles shown are exactly the ones the shipped model predicts for this phrase (verified with `parse()`).

## Reference Material

- `packages/gpu-view/src/features.ts`
- `packages/gpu-view/src/decode.ts`
- `packages/gpu-view/MODEL_CARD.md`
