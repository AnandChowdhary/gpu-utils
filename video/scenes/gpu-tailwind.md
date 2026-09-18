# How gpu-tailwind turns a sentence into utility classes

## Overview

- **Topic**: A 45K-parameter shared-family (ScanTagger) tagger plus a deterministic compiler that turns
  "card with rounded corners, subtle shadow, blue on hover, hidden on mobile" into
  `rounded-lg bg-white p-4 shadow-sm hover:bg-blue-500 max-sm:hidden`, in the browser.
- **Hook**: The model never writes a class name. It only decides which words are
  properties, values, variants and separators; a table compiled from the Tailwind v4
  theme does the rest.
- **Target Audience**: Web developers; no ML or GPU background required.
- **Estimated Length**: 50 seconds, silent, 16:9.
- **Key Insight**: Tagging, not generation: every emitted class is validated against a
  compiled vocabulary, so the output can be wrong but never invalid.

## Narrative Arc

Start with the familiar phrase. Split it into character-class tokens. Show each token
becoming a stack of hashed features. Run the bidirectional scan (accent glow sweeping
left-to-right and right-to-left). Colour the tokens by role. Group them into segments at
the commas and the variant phrase. Compile each segment through the table into classes.
End on the class string reassembled next to the original sentence.

---

## Scene 1: The phrase

**Duration**: ~5 seconds

**Purpose**: Ground the viewer in one familiar input.

### Visual Elements

- Caption: "Natural language"
- Token cells for `card with rounded corners, subtle shadow, blue on hover, hidden on mobile`
  (spaces muted).

### Technical Notes

- Use `Text` with the bundled font, not `Code`.

---

## Scene 2: Sparse features

**Duration**: ~7 seconds

**Purpose**: Show that there is no vocabulary: each token becomes 7 hashed ids.

### Visual Elements

- Caption: "7 hashed features per token, no vocabulary"
- An `activation_strip` under each token.
- Footnote: "word · consonant skeleton · prefix · suffix · shape · length · class"

---

## Scene 3: Bidirectional scan

**Duration**: ~8 seconds

**Purpose**: The only learned part: two gated affine scans (h = a·h + b) sweep the
sequence forward and backward, then a tiny head.

### Visual Elements

- Caption: "Two gated affine scans, 45,394 parameters"
- Accent highlight window sweeping left-to-right, then right-to-left over the strips.
- Footnote: "int6 weights · runs as WGSL compute passes"

---

## Scene 4: Roles

**Duration**: ~8 seconds

**Purpose**: The model's whole output: a role per token and a boundary score.

### Visual Elements

- Caption: "Each token gets a role"
- Token glyphs recoloured: PROPERTY (blue), VALUE (green), VARIANT (amber), SEP (muted).
- Legend line under the tokens.

---

## Scene 5: Segments

**Duration**: ~7 seconds

**Purpose**: Commas and variant phrases split the sentence into groups.

### Visual Elements

- Caption: "Split into segments"
- Thin brackets under `rounded corners` · `subtle shadow` · `blue on hover` · `hidden on mobile`.

---

## Scene 6: Compile

**Duration**: ~9 seconds

**Purpose**: The deterministic compiler maps (property, value) pairs through the Tailwind
v4 table and prefixes variants.

### Visual Elements

- Caption: "Compile through the Tailwind v4 table"
- Under each bracket a class cell appears: `rounded-lg`, `shadow-sm`, `hover:bg-blue-500`,
  `max-sm:hidden`; `bg-white p-4` appear under `card`.
- Footnote: "every class validated against the compiled vocabulary"

---

## Scene 7: Result

**Duration**: ~6 seconds, then a 3 s hold

**Purpose**: Payoff: the sentence and the class string side by side, no server.

### Visual Elements

- Caption: "gpu-tailwind"
- The class cells slide together into one line; sentence above, classes below.
- Footnote: "54.9 KiB Brotli · runs in the browser on WebGPU"

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
| PROPERTY | #60A5FA | model output role |
| VALUE | #34D399 | model output role |
| VARIANT | #FBBF24 | model output role |

## Accuracy Guardrails

- The animation must not imply the model emits class names: classes only appear in the
  compile scene, after the roles.
- It must not imply output is deterministic given the phrase; the roles are predictions.
- Every number shown (45,394 parameters, Brotli size) matches MODEL_CARD.md.

## Reference Material

- `packages/gpu-tailwind/src/features.ts`
- `packages/gpu-tailwind/src/compile.ts`
- `packages/gpu-tailwind/MODEL_CARD.md`
