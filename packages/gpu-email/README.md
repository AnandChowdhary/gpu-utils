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
marker, lines until the next quote-ish line, paragraph index). The summed embeddings go
through six residual dilated 1-D convolutions (kernel 3, dilations 1–32, a 127-token
receptive field) and two heads: a line-kind classifier and a BIO tagger for NAME, TITLE,
COMPANY, PHONE, EMAIL, URL and ADDRESS.

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

The WebGPU path (`src/shader.wgsl`) runs the same embed → 6 × conv → head pipeline with one
thread per (token, channel) and is checked against the PyTorch fixtures on a real adapter
(`training/tests/test_wgsl.py`) and, through them, against the TypeScript CPU path.

## Size and speed

| Measure | Value |
|---|---|
| Parameters | 155,207 (int6, per-tensor scales) |
| Package (min + Brotli, incl. weights) | 92.9 KiB / budget 117.2 KiB (120,000 B) |
| Featurize + CPU forward + decode, 1 KB email (≈530 tokens), Node 24 | ≈ 38 ms |
| Same, 10 KB email (≈5,300 tokens) | ≈ 385 ms |
| WebGPU, 10 KB email (estimate; not measured in CI) | ≈ 5–10 ms warm, 100–300 ms cold |

The budget is 120,000 bytes rather than the 40 KB default because this is a whole-document
dilated-CNN model, not a short-query affine-scan tagger: line and document features need a
2,267-row embedding table and six 48-channel conv layers to cover Gmail, Outlook, Apple
Mail, Thunderbird, mobile and non-English conventions in one model. Most of the size is the
embedding table (109K of the 155K parameters).

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

Held-out generated set: 99.3% line-kind accuracy, 98.9% reply exact match, contact F1
0.96–0.997. Hand-written unfamiliar set (71 emails, styles the generator cannot produce):
89.1% line-kind accuracy, 71.8% reply exact match, contact F1 0.5–0.9. Real fixtures from
email_reply_parser and talon: 28/34 replies exact. See [MODEL_CARD.md](./MODEL_CARD.md).

## Training

```bash
cd packages/gpu-email/training
uv sync
uv run python -m gpu_email.data       # 60K synthetic emails + downloads the eval fixtures
uv run python -m gpu_email.train      # 3 epochs, QAT from 40%, ~16 min on 2 CPU threads
uv run python -m gpu_email.export     # ../model/{manifest.json,weights.txt,fixtures.json}
uv run python -m gpu_email.evaluate   # held-out, unfamiliar and external sets
uv run pytest                         # featurizer, decoder, and WGSL (needs a WebGPU adapter)
```

`src/keywords.ts` is generated from `training/gpu_email/features.py` by
`uv run python -m gpu_email.gen_keywords` (then `pnpm lint:fix`); the keyword tables and
slot layout are shared byte-for-byte and checked by `test/features.test.ts` against row
hashes written by the Python featurizer.
