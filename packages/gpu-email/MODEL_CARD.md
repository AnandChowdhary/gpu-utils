# Model card: gpu-email

## Task
Split plain-text emails into reply, quoted history, and signature, and extract contact
details. Two tagging heads on one trunk: (1) a line-kind classifier over eight kinds
(reply, attribution, quote, signature, disclaimer, forward_header, greeting, closing),
decoded per line by averaging token log-probabilities and smoothing with a Viterbi pass
over sequence constraints; (2) a BIO span tagger for NAME, TITLE, COMPANY, PHONE, EMAIL,
URL and ADDRESS inside the author's signature. Exact rules run first where they are exact:
`>`-prefixed lines are quotes, `-- ` is the signature delimiter, email/URL come from regexes.

## Architecture
- Tokenizer: character-class runs (`@gpu-utils/runtime` `tokenize()`); 37 sparse hashed
  feature ids per token, one fixed-width row (token, line and document-level features; see
  `training/gpu_email/features.py`), no learned vocabulary
- Embedding: 2,267 rows × 48 dims, summed per token (108,816 parameters)
- Sequence mixing: 6 residual dilated 1-D convolutions, kernel 3, dilations 1, 2, 4, 8,
  16, 32, ReLU, zero padding (41,760 parameters; 127-token receptive field)
- Head: shared 48 → 64 ReLU layer, then 64 → 8 line kinds and 64 → 15 BIO labels
  (4,631 parameters)
- Parameters: 155,207 total
- Quantization: int6 symmetric per-tensor (the runtime's text encoding), quantization-aware
  training with a straight-through estimator from 40% of the steps; fixtures and all
  numbers below use the dequantized int6 weights the package ships
- Decoder: per-line mean log-softmax → exact rules → Viterbi with a hand-set 8 × 8
  transition table → segments; BIO Viterbi (O → I forbidden) over the signature block →
  regex override for EMAIL/URL → contact fields

## Training data
- Synthetic, `training/gpu_email/data.py` + `vocab.py`: 60,000 training emails
  (20.8M tokens, seed 1) and 3,000 held-out emails (seed 999). Ten layouts (plain,
  top-post, bottom-post, inline, forward, Outlook chain, mobile, quote-only, short,
  multi-level chain), twelve name locales, 190 titles, generated companies, phone formats
  for 25 countries, addresses for US/UK/IN/AU/CA/DE/FR/ES/JA/ZH, attribution lines in
  English (Gmail, Apple Mail, Thunderbird, Yahoo, GitHub, Outlook `-----Original
  Message-----` and bare `From:/Sent:/To:/Subject:` blocks with bold `*From:*` variants),
  French, German, Spanish, Italian, Dutch, Portuguese, Swedish, Polish, Japanese and
  Chinese, forwarded-message headers (Gmail, Apple, Thunderbird, DE/FR/ES/IT/NL/JA/ZH),
  nested `>`/`> >` and unprefixed Outlook chains, 80 mobile signatures in 12 languages,
  legal disclaimers and mailing-list/GitHub/Google-Groups/Zendesk footers, CRLF, trailing
  whitespace, NBSP, lowercase and wrapped-line noise.
- Real data, evaluation only, never trained on: `github/email_reply_parser` test emails
  (MIT, 22 files) and `mailgun/talon` standard replies (12 `.eml`) and stripped signature
  pairs (6), downloaded by `data.py`. Expected outputs live in
  `training/data/external_expected.json` (line ranges of new content, talon's `Hello`
  replies, talon's signature files). The Enron corpus is not used.
- Unfamiliar set: 71 hand-written emails in `training/data/unfamiliar.json` in styles the
  generator cannot produce (Russian, Korean, Czech, Finnish, Danish, Turkish, Greek and
  Traditional Chinese attributions, Lotus Notes `wrote on` and `|` quoting, full-width
  `＞`, Zendesk/Jira/GitHub/digest notifications, Markdown fences and tables, ASCII-art and
  aligned signatures, `Name / Title / Company`, `Regards, Miguel`, single-initial names,
  `[snip]` bottom posts, body lines starting with `Date:`, out-of-office and calendar
  text, PS after the signature, emoji, tab-indented quotes, `[Quoted text hidden]`,
  `Get Outlook for iOS<https://...>`, CRLF, no trailing newline, ...). Written once, before
  the final evaluation, and not tuned on.

## Evaluation
Metrics use the shipped decoder (rules + Viterbi + regexes), on int6 weights. Line-kind
accuracy is over non-blank lines; reply exact match compares the trimmed reply string;
contact F1 is exact-match per field (set-based for phone/email/url).

__EVAL_TABLES__

The Python (`gpu_email.evaluate`) and TypeScript (`test/eval.test.ts`) decoders are
mirrors; the TypeScript numbers on the unfamiliar set are printed by `pnpm test`.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 93.1 KiB (budget 117.2 KiB) |
| Cold start (device + pipelines + upload) | not measured here (no WebGPU in CI); estimate 100–300 ms |
| Warm call, 1 KB input (≈530 tokens), CPU path, Node 24 | ≈ 38 ms (featurize 2 ms, forward 36 ms, decode < 1 ms) |
| Warm call, 10 KB input (≈5,300 tokens), CPU path | ≈ 385 ms |
| Warm call, WebGPU | estimate 5–10 ms for 10 KB; the `auto` backend switches to WebGPU at 512 tokens |

## Limitations and intended use
Intended for displaying the new part of a message in a thread view, collapsing quoted
history, and pre-filling a contact from a signature. Outputs are probabilistic; validate
contact fields before writing them into a CRM. Known weaknesses: unprefixed quotes whose
attribution has no recognisable marker; one-word lowercase signatures; bare-domain URLs;
disclaimers placed *below* a `-- ` signature (forced to signature by the rule); the
right-to-left and non-Latin coverage is thin (see the unfamiliar-set numbers). Not a
substitute for validation; outputs are probabilistic.

## Checkpoint
- Promoted: __CHECKPOINT__
- Training command: `pnpm train` (`uv run python -m gpu_email.train --minutes 17 --epochs 3`,
  seed 0, 2 CPU threads, `torch.set_num_threads(2)`)
- Data command: `uv run python -m gpu_email.data 60000`
