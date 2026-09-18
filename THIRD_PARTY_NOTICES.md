# Third-party notices

- **Geist Mono** (`video/assets/fonts/GeistMono-Regular.otf`): Copyright (c) Vercel, Inc.
  Licensed under the SIL Open Font License 1.1. https://github.com/vercel/geist-font
- **gpu-lexer** (Shu Ding / Vercel Labs, MIT): engineering reference for the tokenizer →
  sparse features → WGSL tagger recipe and for the explainer video style. No code is copied.
- **gpu-time** (Arik Chakma, MIT): reference for the affine-scan tagger family and the
  render-script conventions. No code is copied.
- **Loghub** (LogPAI, https://github.com/logpai/loghub): the `*_2k.log` samples of 16 systems
  and their `*_structured.csv` field labels are gpu-log's real-world evaluation set.
  `packages/gpu-log/training/gpu_log/loghub.py` maps the CSV columns onto our roles to derive
  gold spans for 16,000 real log lines; `evaluate.py` scores against them. Evaluation only:
  the files are downloaded at evaluation time into `training/data/cache/loghub/`, are never
  committed, and are never used for training or for tuning the generator. Loghub's license
  makes the datasets "freely available for research or academic work" with the request to
  cite: Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. *Loghub: A Large
  Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023.
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

## gpu-tailwind

- [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss) default theme
  (`tailwindcss@4.3.3/theme.css`, Copyright (c) Tailwind Labs, Inc., MIT). Vendored as
  `packages/gpu-tailwind/training/data/tailwind-theme.txt` and compiled into the class
  vocabulary table (`src/table.json`) that ships in the package. Utility naming follows the
  Tailwind v4 documentation; no Tailwind code is included.
- No real NL→Tailwind dataset was used; all training phrases are synthetic
  (`training/gpu_tailwind/data.py`), and the unfamiliar evaluation set is hand-written.
