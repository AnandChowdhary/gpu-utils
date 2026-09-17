# How <package> <does the thing>

## Overview

- **Topic**: 
- **Hook**: 
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 45–60 seconds, silent, 16:9.
- **Key Insight**: 

## Narrative Arc

Start with one familiar input. Decompose it into the units the model sees. Keep those
same units on screen as they become features, flow through the model, and turn into the
typed output. The payoff is the input reassembling as structured output with no server.

---

## Scene 1: <title>

**Duration**: ~5 seconds

**Purpose**: 

### Visual Elements

- 

### Content

### Technical Notes

- Use `Text` with the bundled font, not `Code`.

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

## Accuracy Guardrails

- The animation must not imply <e.g. the model sees field names / that output is deterministic>.
- Every number shown matches MODEL_CARD.md.

## Reference Material

- `packages/<package>/src/features.ts`
- `packages/<package>/MODEL_CARD.md`
