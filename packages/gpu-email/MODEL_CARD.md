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
The shared gpu-utils **conv family**, `ConvTagger(feature_rows=2267, embed=48, hidden=48,
blocks=6, dilations=[1,2,4,8,16,32], tags=23)` from `gpu_utils_training.models`. The package
ships no model code of its own: the CPU reference is the runtime's `convTaggerForward`, the
WebGPU path is `runConvTagger` on the canonical `packages/runtime/src/wgsl/conv_tagger.wgsl`.
Release 0.0.1 used a hand-written two-head `nn.Module` with its own `src/cpu.ts` loop and
`src/shader.wgsl`; the numbers of both are side by side under "Evaluation".

- Tokenizer: character-class runs (`@gpu-utils/runtime` `tokenize()`); 37 sparse hashed
  feature ids per token, one fixed-width row (token, line and document-level features; see
  `training/gpu_email/features.py`), no learned vocabulary
- Embedding: 2,267 rows × 48 dims, summed per token (108,816 parameters)
- Projection: 48 → 48 (2,352 parameters; the family always projects embeddings to `hidden`)
- Sequence mixing: 6 residual dilated blocks `x + relu(conv3_dilated(x)) @ W2 + b2`, kernel
  3, dilations 1, 2, 4, 8, 16, 32, zero padding (55,872 parameters; 127-token receptive
  field). The v1 model had no pointwise `W2` inside a block (41,760 parameters).
- Head: one shared 48 → 48 ReLU layer, then 48 → 23 tag logits (3,479 parameters). The
  first 8 columns are the line-kind classifier, the remaining 15 are the BIO contact
  tagger; `decode.ts` / `decode.py` split them. `pooled_out` is 0 — both heads are
  per-token, so nothing needs the pooled output.
- Parameters: 170,519 total (v1: 155,207; +9.9%, entirely the per-block `W2` and the
  projection, partly offset by a narrower head)
- Quantization: int6 symmetric per-tensor (the runtime's text encoding), quantization-aware
  training with a straight-through estimator from epoch 1 (`gpu_utils_training.qat`);
  fixtures and all numbers below use the dequantized int6 weights the package ships
- Decoder: per-line mean log-softmax → exact rules → Viterbi with a hand-set 8 × 8
  transition table → segments; BIO Viterbi over `bioTransitions` (O → I-X and I-X → I-Y
  forbidden) over the signature block → regex override for EMAIL/URL → contact fields.
  Unchanged from v1 apart from using the runtime's shared Viterbi and BIO transitions.

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

### Held-out, generated (3,000 emails, seed 999; Python decoder)

| Metric | Value |
|---|---|
| Line-kind accuracy (64,377 lines) | **0.9926** |
| per kind: reply / attribution / quote / signature | 0.999 / 1.000 / 0.993 / 0.999 |
| per kind: disclaimer / forward_header / greeting / closing | 0.832 / 0.995 / 0.998 / 0.987 |
| Reply exact match | **0.9887** (2,966 / 3,000) |
| Contact NAME F1 (n=1,535) | 0.995 |
| Contact TITLE F1 (n=1,181) | 0.989 |
| Contact COMPANY F1 (n=1,171) | 0.992 |
| Contact PHONE F1 (n=1,584) | 0.997 |
| Contact EMAIL F1 (n=984) | 0.964 |
| Contact URL F1 (n=755) | 0.955 |
| Contact ADDRESS F1 (n=531) | 0.983 |
| Token-level (training script): line-kind accuracy / BIO F1 | 0.9955 / 0.993 |

Most remaining EMAIL/URL misses are generator noise (a non-breaking space inserted inside
the address, CJK characters in the domain) that the exact regexes reject on purpose. The
TypeScript decoder on the first 500 of the same emails: line-kind 0.9910, reply exact match
0.980, NAME/TITLE/COMPANY/PHONE F1 0.993/0.990/0.993/1.000.

### Unfamiliar, hand-written (71 emails; identical numbers from both decoders)

| Metric | Value |
|---|---|
| Line-kind accuracy (486 lines) | **0.8909** |
| per kind: reply / attribution / quote / signature | 0.812 / 0.893 / 1.000 / 0.891 |
| per kind: disclaimer / forward_header / greeting / closing | 0.538 / 1.000 / 1.000 / 0.833 |
| Reply exact match | **0.7183** (51 / 71) |
| Contact NAME F1 (n=49) | 0.659 (P 0.778, R 0.571) |
| Contact TITLE F1 (n=26) | 0.739 |
| Contact COMPANY F1 (n=25) | 0.717 |
| Contact PHONE F1 (n=23) | 0.913 |
| Contact EMAIL F1 (n=5) | 0.769 |
| Contact URL F1 (n=2) | 0.800 |
| Contact ADDRESS F1 (n=3) | 0.500 |

Top confusions: reply → signature (11), signature → closing (10: one-word name lines such
as "Robert", "Dana", "Tom" read as closings), attribution → signature (5: the Turkish
`-----Özgün İleti-----` header block, no keywords), reply → quote (4: Markdown table rows).
The Zendesk "please type your reply above this line" banner at the top is read as reply.

### External, real emails (evaluation only; 40 emails)

| Set | Metric | Value |
|---|---|---|
| github/email_reply_parser (22) + mailgun/talon standard replies (12) | reply exact match | **0.8235** (28 / 34) |
| mailgun/talon stripped signatures (6) | signature block exact match | 0.333 (2 / 6) |

Reply misses: two bottom-posted "Hello" one-liners directly under a quote with no blank
line (talon apple_mail_2, thunderbird), a bulleted list after two blank lines cut short
(email_bullets), and three long replies where one body line is read as signature. The talon
signature files include the closing line ("Thank you,\nNoam") and separators inside the
signature, which this model labels `closing`; under talon's convention 2 of 6 match, under
this package's kinds 4 of 6 signature blocks are found.

The Python (`gpu_email.evaluate`) and TypeScript (`test/eval.test.ts`) decoders are
mirrors; the TypeScript numbers on the unfamiliar set are printed by `pnpm test`.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 92.9 KiB (budget 117.2 KiB = 120,000 B) |
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
- Promoted: 2026-09-18, seed 0, 3 epochs (8,154 steps, QAT from step 3,261), 15.9 min on 2 threads; `training/runs/latest.pt` (not committed), exported by `gpu_email.export` to `model/`
- Training command: `pnpm train` (`uv run python -m gpu_email.train --minutes 17 --epochs 3`,
  seed 0, 2 CPU threads, `torch.set_num_threads(2)`)
- Data command: `uv run python -m gpu_email.data 60000`
