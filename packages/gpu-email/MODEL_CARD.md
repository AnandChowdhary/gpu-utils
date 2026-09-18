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

**v1** is release 0.0.1's hand-written two-head dilated CNN (155,207 parameters); **v2** is
this release's shared `ConvTagger` (170,519 parameters). Both columns were produced by
`gpu_email.evaluate --exported` on the shipped `model/` artefacts of each release, on the
same three evaluation sets and the same decoder rules, on the same machine. The evaluation
sets did not change.

### Held-out, generated (3,000 emails, seed 999; Python decoder)

| Metric | v1 (custom) | v2 (ConvTagger) |
|---|---|---|
| Line-kind accuracy (64,377 lines) | **0.9926** (63,899) | **0.9926** (63,901) |
| per kind: reply / attribution / quote / signature | 0.999 / 1.000 / 0.993 / 0.999 | 0.999 / 1.000 / 0.993 / 0.998 |
| per kind: disclaimer / forward_header / greeting / closing | 0.832 / 0.995 / 0.998 / 0.987 | 0.842 / 0.998 / 0.999 / 0.987 |
| Reply exact match | **0.9887** (2,966 / 3,000) | **0.9870** (2,961 / 3,000) |
| Contact NAME F1 (n=1,535) | 0.995 | 0.991 |
| Contact TITLE F1 (n=1,181) | 0.989 | 0.992 |
| Contact COMPANY F1 (n=1,171) | 0.992 | 0.990 |
| Contact PHONE F1 (n=1,584) | 0.997 | 0.997 |
| Contact EMAIL F1 (n=984) | 0.964 | 0.962 |
| Contact URL F1 (n=755) | 0.969 | 0.968 |
| Contact ADDRESS F1 (n=531) | 0.983 | 0.981 |

On the generated set v2 is a wash to very slightly worse: line-kind accuracy is unchanged
(two lines better), reply exact match loses five emails of 3,000 (−0.17 pt) and most
contact fields move by ±0.004. Most remaining EMAIL/URL misses in both releases are
generator noise (a non-breaking space inserted inside the address, CJK characters in the
domain) that the exact regexes reject on purpose. The TypeScript decoder on the first 500
of the same emails: line-kind 0.9912, reply exact match 0.980 (v1: 0.9910 / 0.980).

### Unfamiliar, hand-written (71 emails; identical numbers from both decoders)

| Metric | v1 (custom) | v2 (ConvTagger) |
|---|---|---|
| Line-kind accuracy (486 lines) | **0.8909** (433) | **0.9074** (441) |
| per kind: reply / attribution / quote / signature | 0.812 / 0.893 / 1.000 / 0.891 | 0.812 / 0.947 / 1.000 / 0.915 |
| per kind: disclaimer / forward_header / greeting / closing | 0.538 / 1.000 / 1.000 / 0.833 | 0.538 / 1.000 / 1.000 / 0.861 |
| Reply exact match | **0.7183** (51 / 71) | **0.7606** (54 / 71) |
| Contact NAME F1 (n=49) | 0.659 (P 0.778, R 0.571) | 0.706 (P 0.833, R 0.612) |
| Contact TITLE F1 (n=26) | 0.739 | 0.739 |
| Contact COMPANY F1 (n=25) | 0.717 | 0.717 |
| Contact PHONE F1 (n=23) | 0.913 | 0.913 |
| Contact EMAIL F1 (n=5) | 0.769 | 0.909 |
| Contact URL F1 (n=2) | 0.800 | 1.000 |
| Contact ADDRESS F1 (n=3) | 0.500 | 0.571 |

This is where v2 pays off: +1.65 pt line-kind accuracy, +3 replies, and no field regresses.
The attribution kind improves most (0.893 → 0.947: the Turkish `-----Özgün İleti-----` and
Traditional Chinese Outlook header blocks are no longer read as signature).

Remaining confusions: reply → signature (11), signature → closing (7: one-word name lines
such as "Robert", "Dana", "Tom" read as closings), attribution → quote (3), reply → quote
(3: Markdown table rows). The Zendesk "please type your reply above this line" banner at
the top is still read as reply.

### External, real emails (evaluation only; 40 emails)

| Set | Metric | v1 | v2 |
|---|---|---|---|
| github/email_reply_parser (22) + mailgun/talon standard replies (12) | reply exact match | **0.8235** (28 / 34) | **0.8529** (29 / 34) |
| mailgun/talon stripped signatures (6) | signature block exact match | 0.333 (2 / 6) | 0.500 (3 / 6) |

Reply misses: two bottom-posted "Hello" one-liners directly under a quote with no blank
line (talon apple_mail_2, thunderbird), a bulleted list after two blank lines cut short
(email_bullets), and two long replies where one body line is read as signature. The talon
signature files include the closing line ("Thank you,\nNoam") and separators inside the
signature, which this model labels `closing`; under talon's convention 3 of 6 match, under
this package's kinds 5 of 6 signature blocks are found.

### Training-script metric
The training loop's own held-out numbers for the promoted checkpoint are line-kind token
accuracy 0.9956 and BIO **span** F1 0.9764 (`gpu_utils_training.metrics.span_prf` micro).
v1's training script reported 0.9955 and BIO **token** F1 0.993; those two BIO numbers are
different metrics (exact span match versus per-token micro F1) and are not comparable. The
end-to-end contact F1 tables above are the comparable measure.

The Python (`gpu_email.evaluate`) and TypeScript (`test/eval.test.ts`) decoders are
mirrors; the TypeScript numbers on the unfamiliar set are printed by `pnpm test`.

## Size and latency
Both releases measured on the same machine (Node 24, 16-core x86-64 Linux, CPU backend,
median of 30 / 8 runs on the same generated emails).

| Measure | v1 (custom) | v2 (ConvTagger) |
|---|---|---|
| Parameters | 155,207 | 170,519 (+9.9%) |
| Package (min + Brotli, incl. weights) | 93.0 KiB | 108.9 KiB (+17%, budget 117.2 KiB = 120,000 B) |
| Warm call, 1 KB input (459 tokens), CPU path | 32.7 ms | 39.1 ms (+20%) |
| Warm call, 10 KB input (4,116 tokens), CPU path | 298 ms | 355 ms (+19%) |
| Cold start (device + pipelines + upload) | not measured here (no WebGPU in CI); estimate 100–300 ms ||
| Warm call, WebGPU | estimate 5–10 ms for 10 KB; the `auto` backend switches to WebGPU at 512 tokens ||

The size and latency cost is the shared conv block's extra pointwise `W2` (48 × 48 per
block, six blocks) plus the 48 × 48 input projection: +15,312 parameters and one extra
matmul per token per block. It buys the canonical kernel and CPU forward pass, which are
parity-tested on every CI run, and the better unfamiliar-set numbers above. The package
stays under its 120,000-byte budget with 11.1 KiB of headroom.

## Limitations and intended use
Intended for displaying the new part of a message in a thread view, collapsing quoted
history, and pre-filling a contact from a signature. Outputs are probabilistic; validate
contact fields before writing them into a CRM. Known weaknesses: unprefixed quotes whose
attribution has no recognisable marker; one-word lowercase signatures; bare-domain URLs;
disclaimers placed *below* a `-- ` signature (forced to signature by the rule); the
right-to-left and non-Latin coverage is thin (see the unfamiliar-set numbers). Not a
substitute for validation; outputs are probabilistic.

## Checkpoint
- Promoted: 2026-09-18, seed 0, 3 epochs (8,156 steps, int6 QAT from epoch 1), 16.45 min
  on 2 CPU threads; `training/runs/default/best.pt` (not committed), exported by
  `gpu_email.export` to `model/`. Held-out selection metrics at the promoted epoch:
  line-kind token accuracy 0.9956, BIO span F1 0.9764.
- Training command: `pnpm train` (`uv run python -m gpu_email.train`, defaults: 3 epochs,
  lr 2.5e-3, batch 24, 18-minute budget, seed 0, 2 CPU threads)
- Export command: `pnpm export` (`uv run python -m gpu_email.export`)
- Data command: `uv run python -m gpu_email.data 60000`
