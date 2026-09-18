# Third-party notices

- **Geist Mono** (`video/assets/fonts/GeistMono-Regular.otf`): Copyright (c) Vercel, Inc.
  Licensed under the SIL Open Font License 1.1. https://github.com/vercel/geist-font
- **gpu-lexer** (Shu Ding / Vercel Labs, MIT): engineering reference for the tokenizer →
  sparse features → WGSL tagger recipe and for the explainer video style. No code is copied.
- **gpu-time** (Arik Chakma, MIT): reference for the affine-scan tagger family and the
  render-script conventions. No code is copied.
- **email_reply_parser** (GitHub, MIT): `test/emails/*.txt` fixtures are downloaded at
  evaluation time by `packages/gpu-email/training/gpu_email/data.py` and used only to
  evaluate gpu-email (reply extraction). Not committed, not used for training.
  https://github.com/github/email_reply_parser
- **talon** (Mailgun, Apache License 2.0): `tests/fixtures/standard_replies/*.eml` and
  `tests/fixtures/signature/emails/stripped/*` are downloaded at evaluation time by the
  same script and used only to evaluate gpu-email (reply and signature extraction). Not
  committed, not used for training. https://github.com/mailgun/talon
