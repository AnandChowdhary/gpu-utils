# Third-party notices

- **Geist Mono** (`video/assets/fonts/GeistMono-Regular.otf`): Copyright (c) Vercel, Inc.
  Licensed under the SIL Open Font License 1.1. https://github.com/vercel/geist-font
- **gpu-lexer** (Shu Ding / Vercel Labs, MIT): engineering reference for the tokenizer →
  sparse features → WGSL tagger recipe and for the explainer video style. No code is copied.
- **gpu-time** (Arik Chakma, MIT): reference for the affine-scan tagger family and the
  render-script conventions. No code is copied.
- **anystyle core dataset** (`res/parser/core.xml`, Sylvester Keil, BSD-2-Clause,
  https://github.com/inukshuk/anystyle): 1,514 hand-labelled reference strings used by
  `packages/gpu-cite` for evaluation only. Downloaded at training time, not committed.
- **GROBID citation training corpus** (`grobid-trainer/resources/dataset/citation`, Apache-2.0,
  https://github.com/grobidOrg/grobid): ~3,600 TEI-annotated reference strings used by
  `packages/gpu-cite` for evaluation only. Downloaded at training time, not committed.
- **CrossRef REST API metadata** (https://api.crossref.org, metadata is CC0/public domain per
  CrossRef's terms) and **arXiv OAI-PMH metadata** (https://oaipmh.arxiv.org, CC0): structured
  bibliographic records used by `packages/gpu-cite` as input to its synthetic reference
  renderer. Downloaded at training time, not committed.
