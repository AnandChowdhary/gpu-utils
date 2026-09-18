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

## gpu-paste (training data only; nothing below ships in the package)

- [Faker](https://github.com/joke2k/faker) — MIT. Names, streets, cities, postcodes and
  company names in 30+ locales drive the synthetic generator (`training/gpu_paste/data.py`).
- [US Census Bureau 1990 name files](https://www.census.gov/topics/population/genealogy/data/1990_census/1990_census_namefiles.html)
  (`dist.all.last`, `dist.female.first`, `dist.male.first`) — public domain (US federal
  government work). Downloaded at training time into `training/data/cache/`, not committed.
- [SEC EDGAR `company_tickers.json`](https://www.sec.gov/file/company-tickers) — public
  domain. Downloaded at training time for real company names, not committed.
- Considered and not used: OpenAddresses (per-source licences, several share-alike),
  libpostal / gpu-postal training corpora (OSM-derived, ODbL), CoNLL-style NER corpora
  (research-only licences).
- **email_reply_parser** (GitHub, MIT): `test/emails/*.txt` fixtures are downloaded at
  evaluation time by `packages/gpu-email/training/gpu_email/data.py` and used only to
  evaluate gpu-email (reply extraction). Not committed, not used for training.
  https://github.com/github/email_reply_parser
- **talon** (Mailgun, Apache License 2.0): `tests/fixtures/standard_replies/*.eml` and
  `tests/fixtures/signature/emails/stripped/*` are downloaded at evaluation time by the
  same script and used only to evaluate gpu-email (reply and signature extraction). Not
  committed, not used for training. https://github.com/mailgun/talon
