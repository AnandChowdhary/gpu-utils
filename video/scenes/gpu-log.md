# How gpu-log reads any log line

## Overview

- **Topic**: One raw log line becoming a typed record (timestamp, level, source, thread, message, key=value pairs) without a format-specific parser.
- **Hook**: A log4j line, a syslog line and a Node stack frame look nothing alike, yet the same tiny model labels all three token by token.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 45–60 seconds, silent, 16:9.
- **Key Insight**: Logs are made of the same handful of roles in different orders. A character-class tokenizer plus hashed features gives a 150K-parameter dilated CNN enough context to tag every token; a deterministic compiler does the rest, and thousands of lines ride one GPU dispatch.

## Narrative Arc

Start with one familiar log4j line. Split it into character-class tokens that stay on
screen as pills. Show each pill becoming sparse hashed features (no vocabulary), then the
five dilated convolution windows widening around one token. Tags land on the pills in
semantic colours; the compiler folds them into the typed record. Pull back to a wall of
lines to show the batch-per-dispatch design, and end on the record with the size and speed.

---

## Scene 1: A line you have seen before

**Duration**: ~6 seconds

**Purpose**: Anchor on a real-looking log4j line.

### Visual Elements

- Caption: "A log line"
- The line `2024-01-15 10:30:00,123 [main] INFO com.example.Foo - Started in 12ms` in mono, white.

### Technical Notes

- Use `Text` with the bundled font, not `Code`.

---

## Scene 2: Character-class tokens

**Duration**: ~7 seconds

**Purpose**: Show the mechanical tokenizer: runs of letters, digits, spaces, single punctuation.

### Visual Elements

- Caption: "Split into character-class runs"
- The line becomes `TokenCell` pills (spaces muted), same glyphs, same order.
- Footnote: "no vocabulary, no regexes per format"

---

## Scene 3: Sparse hashed features

**Duration**: ~7 seconds

**Purpose**: Each token gets nine hashed ids (word, prefix, shape, first/last char, length, position, column).

### Visual Elements

- Caption: "9 hashed feature ids per token"
- `activation_strip` under each pill in the accent colour.

---

## Scene 4: Dilated convolutions

**Duration**: ~9 seconds

**Purpose**: Show context growing: five residual blocks with dilations 1, 2, 4, 8, 16.

### Visual Elements

- Caption: "5 dilated conv blocks, dilation 1 → 16"
- Accent brackets widening around the `INFO` pill: ±1, ±3, ±7, ±15, ±31 tokens.
- Footnote: "150K parameters, int6"

---

## Scene 5: Tags

**Duration**: ~8 seconds

**Purpose**: The model's only output: one BIO role per token plus a line kind.

### Visual Elements

- Caption: "One role per token"
- Pills recolour by role: TS, THREAD, LEVEL, SOURCE, MSG; spaces and separators stay muted.
- Small label "entry" beside the line.

---

## Scene 6: The compiler

**Duration**: ~8 seconds

**Purpose**: Deterministic TypeScript turns tags into the typed record and normalises values.

### Visual Elements

- Caption: "Compiled into a typed record"
- Pills `ReplacementTransform` into the JSON-ish record: timestamp (ISO), level "info", source, thread, message.

---

## Scene 7: Whole files per dispatch

**Duration**: ~7 seconds

**Purpose**: Throughput comes from batching: lines are packed back to back with line ids so one dispatch tags thousands of lines.

### Visual Elements

- Caption: "Thousands of lines per GPU dispatch"
- A column of muted lines with accent line-id ticks; JSON and logfmt lines skip the model (marked "deterministic").

---

## Scene 8: Hold

**Duration**: ~5 seconds

**Purpose**: Land the numbers.

### Visual Elements

- Caption: "gpu-log"
- Record stays; footnote with parameters, Brotli size and the CPU throughput from MODEL_CARD.md.

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
| Accent | #3B82F6 | activations, conv windows |
| TS | #22C55E | model output |
| LEVEL | #F59E0B | model output |
| SOURCE | #A855F7 | model output |
| THREAD | #06B6D4 | model output |
| MSG | #F43F5E | model output |

## Accuracy Guardrails

- The animation must not imply the model sees field names, knows formats, or generates text: it only tags tokens.
- The record shown is what `parse()` returns for the line; ISO conversion is done by the compiler, not the model.
- Every number shown (parameters, size, throughput) matches MODEL_CARD.md.

## Reference Material

- `packages/gpu-log/src/features.ts`
- `packages/gpu-log/src/decode.ts`
- `packages/gpu-log/MODEL_CARD.md`
