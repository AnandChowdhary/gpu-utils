# How gpu-email splits an email into reply, quote and signature

## Overview

- **Topic**: gpu-email turns a plain-text email body into labelled line segments (reply,
  attribution, quote, signature, disclaimer, forward header, greeting, closing), the new
  reply text, and the author's contact fields.
- **Hook**: an email thread is a stack of messages glued together by mail clients in a
  dozen conventions; a 155K-parameter tagger sorts the lines in a few milliseconds.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 50–60 seconds, silent, 16:9.
- **Key Insight**: every token carries features about its *line* and its *document
  position*, so a dilated 1-D CNN sees enough context to label lines, and a small set of
  exact rules (`>` prefixes, `-- `) plus a Viterbi pass over line kinds cleans up the rest.

## Narrative Arc

Start with one familiar top-posted reply. Split it into lines, then into tokens. Show the
sparse feature ids each token receives (word hash, shape, line-level flags, distance to
the attribution line). Send the same token cells through six dilated convolutions whose
receptive field visibly widens. Read two heads off the trunk: one paints each line with a
kind, the other underlines name / title / phone / email inside the signature. Finish with
the typed output: `reply`, `segments`, `contact`.

---

## Scene 1: A reply with history

**Duration**: ~6 seconds

**Purpose**: Establish the input.

### Visual Elements

- Caption: "One email, three messages glued together"
- Twelve lines of an email in white (greeting, body, closing, four-line signature,
  attribution line, three quoted lines), left-aligned monospace.

### Technical Notes

- Use `Text` with the bundled font, not `Code`. Lines appear with `LaggedStart`.

---

## Scene 2: Lines become tokens

**Duration**: ~7 seconds

**Purpose**: Show the mechanical tokenizer: character-class runs, no vocabulary.

### Visual Elements

- Caption: "Split into character-class runs"
- Three representative lines ("Best,", "John Doe", "> Can we move it?") explode into
  `TokenCell`s; whitespace cells muted.
- Footnote: "no vocabulary, no server"

---

## Scene 3: Sparse features per token

**Duration**: ~8 seconds

**Purpose**: Explain the hashed features, especially the line-level and document-level ones.

### Visual Elements

- Caption: "37 hashed feature ids per token"
- Under each cell an `activation_strip` in the accent color.
- Three labelled feature groups fade in as muted text: "word · shape · suffix",
  "line: first word, ends with ':', looks like a phone", "document: quote lines above,
  lines since 'wrote:'".

### Accuracy note

- Do not imply the model reads field names; it only sees hashed ids.

---

## Scene 4: Dilated convolutions widen the view

**Duration**: ~9 seconds

**Purpose**: Show the trunk: six residual 1-D convolutions with dilation 1, 2, 4, 8, 16, 32.

### Visual Elements

- Caption: "Six dilated convolutions, kernel 3"
- The token strip stays; above it, six rows of accent bars connect each position to
  neighbours at growing offsets (1, 2, 4, 8, 16, 32), drawn as short arcs.
- Footnote: "receptive field: 127 tokens · 155K parameters · int6"

---

## Scene 5: Two heads

**Duration**: ~9 seconds

**Purpose**: The line-kind head and the BIO contact head read the same trunk state.

### Visual Elements

- Caption: "Head 1 paints lines, head 2 tags contact fields"
- The email lines from Scene 1 re-appear; each line's background tints with its kind
  (semantic colors: reply, closing, signature, attribution, quote).
- Inside the signature, underline spans labelled NAME, TITLE, PHONE, EMAIL.

---

## Scene 6: Rules and Viterbi clean up

**Duration**: ~7 seconds

**Purpose**: Exact rules first, then sequence smoothing.

### Visual Elements

- Caption: "Exact rules first, then Viterbi over line kinds"
- The `>` lines flash: "prefix '>' → quote (rule)". The attribution → quote transition
  highlighted as an arrow; a wrongly-tinted body line snaps to its neighbours' colour.

---

## Scene 7: Typed output

**Duration**: ~8 seconds + 3 s hold

**Purpose**: Payoff.

### Visual Elements

- Caption: "reply · segments · contact"
- Right side: a small JSON-like block with `reply: "Thanks for the update…"`,
  `segments: [greeting, reply, closing, signature, attribution, quote]`,
  `contact: { name: "John Doe", title: "CEO", phone: [...], email: [...] }`.
- Footnote with real numbers from MODEL_CARD.md: package size and warm latency.

---

## Recurring Visual Motifs

- Token cells (`TokenCell`) persist from decomposition to final labels.
- Accent `#3B82F6` marks learned state only (feature strips, conv arcs).
- Semantic colors appear only in Scene 5–7 for the model's outputs.

## Color Palette

| Role | Hex | Usage |
|---|---|---|
| Background | #000000 | frame |
| Text | #F5F5F5 | primary |
| Muted | #999999 | footnotes, whitespace glyphs |
| Accent | #3B82F6 | activations, conv arcs |
| reply | #F5F5F5 | line tint (kept white) |
| quote | #6B7280 | line tint |
| signature | #22C55E | line tint / field underline |
| attribution | #F59E0B | line tint |
| closing / greeting | #A78BFA | line tint |

## Accuracy Guardrails

- The animation must not imply the model sees field names, regexes, or a vocabulary;
  it sees hashed ids only. Rules (`>` prefix, `-- `, email/URL regex) are shown as
  explicit "rule" steps, separate from the model.
- The animation must not imply the output is deterministic in all cases; the footnote
  says "probabilistic tagger + exact rules".
- Every number shown (155K parameters, six layers, dilations, receptive field 127,
  package size, latency) matches MODEL_CARD.md.

## Reference Material

- `packages/gpu-email/src/features.ts`
- `packages/gpu-email/src/decode.ts`
- `packages/gpu-email/MODEL_CARD.md`
