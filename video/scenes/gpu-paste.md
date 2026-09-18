# How gpu-paste understands a paste

## Overview

- **Topic**: What happens between Ctrl-V and `{ kind: "contact", ... }`.
- **Hook**: A pasted signature block turns into typed fields with no server involved.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 50–60 seconds, silent, 16:9.
- **Key Insight**: Rules decide everything that can be validated; a 66K-parameter model
  only does the two things rules cannot: pick the ambiguous kind and tag entities.

## Narrative Arc

Start with one familiar paste (a four-line signature). Split it into the tokenizer's cells.
A regex claims the email immediately (rule, white). The remaining cells grow sparse
feature strips, a scan sweeps across them both ways, and the same cells light up with span
labels (semantic colors). The pooled context picks "contact". The cells reassemble as the
typed result.

---

## Scene 1: A paste

**Duration**: ~6 seconds

**Purpose**: Establish the input as ordinary clipboard text.

### Visual Elements

- Caption "Pasted text".
- Four lines of `TokenCell`s: `Jane Doe`, `Acme Corp`, `+1 555 0100`, `jane@acme.com`.

---

## Scene 2: Rules first

**Duration**: ~7 seconds

**Purpose**: Show the deterministic layer claiming what it can validate.

### Visual Elements

- Caption "Rules first: what can be validated is never guessed".
- The email cells get a white `email · rule` tag. Footnote: "JSON, CSV, URL, UUID, IP,
  color, money, dates… all rules".

---

## Scene 3: Sparse features

**Duration**: ~7 seconds

**Purpose**: Tokens become hashed feature ids, no vocabulary.

### Visual Elements

- Caption "Each token → 10 hashed features".
- `activation_strip` under every remaining cell (accent).

---

## Scene 4: Bidirectional scan

**Duration**: ~8 seconds

**Purpose**: Context flows across the paste in both directions.

### Visual Elements

- Caption "Gated affine scan, both directions".
- An accent bar sweeps left→right, then right→left over the cells.

---

## Scene 5: Span head

**Duration**: ~8 seconds

**Purpose**: The model tags entities the rules could not.

### Visual Elements

- Caption "Span head: person · company · phone".
- Cells recolor by span kind (semantic colors appear here for the first time).

---

## Scene 6: Kind head

**Duration**: ~7 seconds

**Purpose**: Mean-pooled context decides the ambiguous kind.

### Visual Elements

- Caption "Mean-pool → kind".
- A bracket under all cells collapses into `contact 0.99`.

---

## Scene 7: Typed output

**Duration**: ~8 seconds

**Purpose**: The payoff: structured fields.

### Visual Elements

- Caption "parse(text) →".
- `{ kind: "contact", name: "Jane Doe", company: "Acme Corp", phone: "+15550100", email: "jane@acme.com" }`

---

## Scene 8: Hold

**Duration**: ~4 seconds

- Footnote with real numbers from MODEL_CARD.md: parameters, Brotli size, "no server".

---

## Recurring Visual Motifs

- Token cells (`TokenCell`) persist from decomposition to final labels.
- Accent `#3B82F6` marks learned state only.

## Color Palette

| Role | Hex | Usage |
|---|---|---|
| Background | #000000 | frame |
| Text | #F5F5F5 | primary, rule tags |
| Muted | #999999 | footnotes, whitespace glyphs |
| Accent | #3B82F6 | activations, scan windows |
| person | #F59E0B | span label |
| company | #10B981 | span label |
| phone | #EC4899 | span label |

## Accuracy Guardrails

- The animation must not imply the model reads the email: the email is a regex hit.
- The animation must not imply a learned vocabulary: features are hashes.
- Every number shown matches MODEL_CARD.md (parameters, Brotli size).

## Reference Material

- `packages/gpu-paste/src/features.ts`
- `packages/gpu-paste/src/rules.ts`
- `packages/gpu-paste/MODEL_CARD.md`
