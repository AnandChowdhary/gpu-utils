# gpu-email

Split plain-text emails into reply, quoted history, and signature, and extract contact details.

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-email
```

```ts
import { parse } from "gpu-email";

const result = await parse(`Hi Bob,

Thanks for the update, that works for me.

Best,
John Doe
CEO, Acme Inc
+1 (555) 123-4567
john@acme.com

On Mon, Jan 5, 2024 at 3:14 PM Bob Smith <bob@example.com> wrote:
> Hi John,
> Can we move the meeting?
`);

result.reply;
// "Hi Bob,\n\nThanks for the update, that works for me.\n\nBest,"
result.segments.map((s) => s.kind);
// ["greeting", "reply", "closing", "signature", "attribution", "quote"]
result.contact;
// { name: "John Doe", title: "CEO", company: "Acme Inc",
//   phone: ["+1 (555) 123-4567"], email: ["john@acme.com"], span: [56, 118] }
```

The input is one plain-text body (after MIME decoding; no HTML). The output is

```ts
{
  segments: { kind: "reply" | "attribution" | "quote" | "signature" | "disclaimer"
                  | "forward_header" | "greeting" | "closing";
              span: [start, end] }[];   // UTF-16 offsets, half-open, in document order
  reply: string;                        // new content only (reply + greeting + closing), trimmed
  contact?: { name?, title?, company?, phone?: string[], email?: string[], url?: string[],
              address?, span: [start, end] };   // from the author's own signature block
  diagnostics: { backend: "cpu" | "webgpu"; tokens; lines; ms };
}
```

`parse(text, { backend })` accepts `"auto"` (default: WebGPU for inputs of 512+ tokens when
available, CPU otherwise), `"webgpu"` or `"cpu"`. `parseMany(texts)` batches calls. When
WebGPU is missing the CPU reference path is used; it never returns empty results silently.

## How it works

The text is split into character-class runs by the shared gpu-utils tokenizer, and every
token gets 37 hashed feature ids: a few about the token (word hash, shape, suffix, position
in its line), most about its *line* (first/last word hash, trailing punctuation, word and
digit counts, title-case ratio, whether it looks like an email/URL/phone, keyword hits for
"wrote"/"a écrit"/"schrieb"/"escribió"/"写道", "Original Message", "Forwarded message",
"Sent from my", header keys such as `From:`/`Von:`/`De :`/`差出人:`, legal-disclaimer words,
greeting and closing words, month and weekday names in six languages) and a handful about
the *document* (any `>` line above, a `-- ` line above, lines since the last attribution
marker, lines until the next quote-ish line, paragraph index). The model is the shared
gpu-utils *conv family* (`ConvTagger`): the summed embeddings are projected to 48 channels
and go through six residual dilated 1-D convolution blocks (kernel 3, dilations 1–32, a
127-token receptive field), then one per-token head whose 23 logits are read as two
outputs: 8 line-kind columns and 15 BIO columns for NAME, TITLE, COMPANY, PHONE, EMAIL,
URL and ADDRESS.

Decoding is rules first where a rule is exact, model for the rest: lines that start with
`>` are quotes, a line that is exactly `--`/`-- ` is the RFC 3676 signature delimiter and
everything below it is signature until an attribution, forward or header marker line, and
email addresses and URLs inside the signature come from regular expressions rather than the
tagger. Everything else is the model: token log-probabilities are averaged per line, then a
Viterbi pass with a hand-set transition table (attribution → quote is free, quote → reply
costs a little so inline replies survive, signature → reply is expensive, ...) picks one
kind per line. Consecutive lines of one kind become a segment; `reply` is the reply,
greeting and closing lines joined; `contact` is BIO-decoded from the first signature block
that follows the author's own text.

Both forward passes come from `@gpu-utils/runtime`: the CPU path is `convTaggerForward`
(the reference) and the WebGPU path is `runConvTagger` on the canonical `conv_tagger.wgsl`
kernel (one workgroup per token, multi-pass). The kernel is checked against the PyTorch
fixtures on a real adapter (`training/tests/test_wgsl.py`, Mesa lavapipe in CI) and, through
them, against the TypeScript CPU path (`test/parity.test.ts`). This package has no model
code of its own beyond the featurizer and the decoder.

## Size and speed

| Measure | Value |
|---|---|
| Parameters | 170,519 (int6, per-tensor scales) |
| Package (min + Brotli, incl. weights) | SIZE_TBD KiB / budget 117.2 KiB (120,000 B) |
| Featurize + CPU forward + decode, 1 KB email (≈530 tokens), Node 24 | ≈ LAT1_TBD ms |
| Same, 10 KB email (≈5,300 tokens) | ≈ LAT10_TBD ms |
| WebGPU, 10 KB email (estimate; not measured in CI) | ≈ 5–10 ms warm, 100–300 ms cold |

The budget is 120,000 bytes rather than the 40 KB default because this is a whole-document
conv-family model, not a short-query scan-family tagger: line and document features need a
2,267-row embedding table and six 48-channel residual blocks to cover Gmail, Outlook, Apple
Mail, Thunderbird, mobile and non-English conventions in one model. Most of the size is the
embedding table (109K of the 171K parameters).

## Limitations

- Plain text only. HTML must be converted first; `[image: ...]` placeholders are fine.
- Quote detection without `>` prefixes (Outlook, Lotus Notes `|`, Japanese `＞`) is
  learned, not exact; a quoted block whose attribution line has no recognisable marker
  (e.g. a language the keyword lists do not cover) can be read as reply text.
- Contact extraction only reads the author's own signature (the first signature block after
  the author's text). Bare domains without `www.` or a scheme are not URLs; phone numbers
  need at least five digits; single lowercase words ("rick") are often not recognised as
  names.
- Greetings and closings are counted as reply text. A "PS" after the signature is reply text
  only if the model says so; the `-- ` rule forbids it.
- Non-Latin scripts are covered by the featurizer (it is Unicode-aware), but the training
  data is mostly English with French, German, Spanish, Italian, Dutch, Portuguese, Swedish,
  Polish, Japanese and Chinese; see MODEL_CARD.md for the unfamiliar-set numbers on Korean,
  Czech, Finnish, Danish, Turkish, Greek and Traditional Chinese.

EVAL_SUMMARY_TBD See [MODEL_CARD.md](./MODEL_CARD.md) for the full tables, including the
numbers of the previous custom-model release side by side.

## Training

```bash
cd packages/gpu-email/training
uv sync
uv run python -m gpu_email.data       # 60K synthetic emails + downloads the eval fixtures
uv run python -m gpu_email.train      # 2 epochs, int6 QAT from epoch 1, 18 min budget on 2 CPU threads
uv run python -m gpu_email.export     # ../model/{manifest.json,weights.txt,fixtures.json} + row hashes
uv run python -m gpu_email.evaluate   # held-out, unfamiliar and external sets (--exported for ../model)
uv run pytest                         # featurizer, decoder, model fixtures, and WGSL (needs a WebGPU adapter)
```

The model, training loop, quantization, metrics, Viterbi, export and WGSL harness all come
from `gpu_utils_training` (`tooling/python`); `training/gpu_email/` only holds the
featurizer, the generator, the decoder mirror and the evaluation sets.

`src/keywords.ts` is generated from `training/gpu_email/features.py` by
`uv run python -m gpu_email.gen_keywords` (then `pnpm lint:fix`); the keyword tables and
slot layout are shared byte-for-byte and checked by `test/features.test.ts` against row
hashes written by the Python featurizer.
