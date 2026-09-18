# gpu-paste

Understand pasted text: detect what it is and extract structured fields, on-device.

Tiny model, trained from scratch, runs on WebGPU in the browser. Zero dependencies.
Part of [gpu-utils](https://github.com/AnandChowdhary/gpu-utils).

```bash
npm install gpu-paste
```

```ts
import { parse } from "gpu-paste";

const result = await parse(`Jane Doe
Senior Engineer, Acme Corp
+1 (555) 123-4567
jane@acme.com`);

result.kind; // "contact"
result.confidence; // 0.99
result.spans;
// [
//   { kind: "person",  span: [0, 8],   value: "Jane Doe" },
//   { kind: "company", span: [26, 35], value: "Acme Corp" },
//   { kind: "phone",   span: [36, 53], value: "+15551234567" },
//   { kind: "email",   span: [54, 67], value: "jane@acme.com" },
// ]
result.parsed;
// { name: "Jane Doe", company: "Acme Corp", email: "jane@acme.com", phone: "+15551234567" }
result.diagnostics; // { decidedBy: "model", backend: "cpu", tokens: 34, kindProbabilities: {...}, ms: 0.8 }
```

`parse(text, { backend?: "auto" | "webgpu" | "cpu" })` returns:

| Field | Meaning |
|---|---|
| `kind` | one of `json` `csv` `tsv` `markdown` `html` `code` `url` `email` `phone` `address` `contact` `datetime` `color` `uuid` `jwt` `ip` `path` `money` `number` `list` `prose` `empty` |
| `confidence` | 1.0 for validated rules, 0.6–0.95 for heuristic rules, softmax probability for learned kinds |
| `spans` | `{ kind, span: [start, end], value? }` in UTF-16 code units, half-open; kinds `email` `phone` `url` `date` `money` `address` `person` `company` `color` `ip` `uuid` `hashtag` `mention` `issue_ref` `commit` |
| `parsed` | `JSON.parse` result, `{ delimiter, header?, rows }` for CSV/TSV, `{ hex, rgb, alpha? }` for colors, `{ amount, currency }` for money, a `number`, `{ iso?, raw }` for dates, `{ name, company, email, phone, url, address }` for contacts, `{ lines }` for addresses, `{ items }` for lists, `{ headings, links }` for markdown, URL parts, `{ header, payload }` for JWTs, … |
| `diagnostics` | what decided the kind (`rule:json`, `model`, `model+promotion`), backend, token count, kind probabilities, notes (e.g. "day/month order ambiguous") |

Under `"auto"`, inputs under 256 tokens use the CPU reference path (GPU readback dominates
below that); WebGPU is used for longer pastes when available, otherwise the CPU path is used.
No result is ever empty because WebGPU is missing.

## How it works

1. **Rules first.** `src/rules.ts` tries every detector that can be *validated*:
   `JSON.parse`, a CSV/TSV sniffer (consistent field counts, RFC 4180 quoting), balanced
   HTML tags, `new URL()`, e-mail, UUID (with version), JWT (base64url header with `alg`),
   IPv4/IPv6 with range checks, `#hex`/`rgb()`/`hsl()`/named colors, file paths, money in
   ~40 currency symbols/codes and locale number formats, ISO/RFC/numeric/month-name dates,
   numbers. A hit ends the kind decision; single-entity kinds are also the paste's one span.
2. **Tokenize + features.** `tokenize()` from `@gpu-utils/runtime` splits the text into
   character-class runs. Each token gets 10 hashed ids (word, consonant skeleton, shape,
   two-char prefix/suffix, length bucket, column in line, line from start, line from end,
   bias) that index one 1,575-row × 32-dim embedding table. No vocabulary is learned.
3. **Model.** Summed embeddings → two gated affine scans (`h = a·h₋₁ + (1−a)·u`, forward
   and backward, run as a parallel prefix scan in WGSL) → pointwise mix (96→48) → mean-pooled
   context. A BIO head tags `person` / `company` / `address` / `date` / `money` / `phone`
   per token; a kind head on the pooled vector picks `address` / `contact` / `prose` /
   `list` / `code` / `markdown`. 66,339 parameters, int6.
4. **Decode.** Constrained Viterbi over the BIO tags → character spans, merged with the
   regex spans (rules win on overlap). If the model says `prose` but a single learned span
   covers ≥ 90 % of the paste, the kind is promoted deterministically (`date` → `datetime`,
   `money` → `money`, `phone` → `phone`, `person`/`company` → `contact`). `parsed` is compiled
   from the spans and the rule parsers.

### Rules vs. learned

| Decision | How | Where |
|---|---|---|
| `empty`, `json`, `csv`, `tsv`, `html`, `url`, `email`, `uuid`, `jwt`, `ip`, `color`, `path`, `number` | deterministic, validated | `rules.ts` `detectWhole()` |
| `phone`, `datetime`, `money` as a whole paste | deterministic regex + parser (confidence < 1) | `rules.ts` |
| `address`, `contact`, `prose`, `list`, `code`, `markdown` | learned kind head (6-way softmax) | `cpu.ts` / `shader.wgsl` |
| spans `email`, `url`, `uuid`, `ip`, `color`, ISO `date`, `hashtag`, `mention`, `issue_ref`, `commit` | deterministic regex | `rules.ts` `ruleSpans()` |
| spans `person`, `company`, `address`, `date` phrases, `money` in any locale, `phone` in any format | learned BIO head + Viterbi | `decode.ts` |
| span `value` normalisation (E.164-ish phones, ISO dates, `amount currency`) | deterministic parsers | `rules.ts` |
| single-entity kind promotion, `parsed` for contact/address/list/markdown | deterministic compiler | `index.ts` |

The model never sees or decides anything in the first, second or fourth rows. A CSV of
contacts is still `csv`; its e-mails come from the regex and its names from the model.

## Size and speed

| Measure | Value |
|---|---|
| Package (min + Brotli, incl. int6 weights and WGSL) | 51.0 KiB (52,224 B); budget 60,000 B |
| Weights alone (66,339 int6 params, Brotli) | ~38 KB |
| Parameters | 66,339 |
| CPU path, short contact (12 tokens), Node 24 | 0.5 ms warm |
| CPU path, 1 KB paste (425 tokens), Node 24 | 13–14 ms warm (≈ 30 µs / token) |
| WebGPU cold start (device + 7 pipelines + weight upload) | ≈ 50–100 ms (estimate, no GPU on the build box) |
| WebGPU warm call, 1 KB paste | ≈ 1–3 ms, dominated by readback (estimate) |

The budget is 60 KB rather than the default 40 KB because the package also ships ~14 KB
(Brotli) of deterministic parsers: 148 CSS color names, ~40 currencies, date grammars in
four languages, CSV/HTML/path/IP/JWT validation. See [MODEL_CARD.md](./MODEL_CARD.md) for
the measured number.

## Limitations

- Kind detection is single-label. A markdown document that is mostly a list may come back
  as `list`; a signature under three paragraphs of e-mail is `prose` with contact spans.
- CSV sniffing needs ≥ 2 lines with identical field counts; "milk, eggs\nbread, butter"
  is therefore `csv`, not `list`. Markdown tables (`| a | b |`) are excluded.
- JSX/TSX and HTML templates with a matched tag pair are `html`, not `code`.
- Bare 10–11 digit numbers are guessed as `phone` at 0.6 confidence; everything else
  numeric is `number`.
- `MM/DD` vs `DD/MM` is guessed (day-first for `.`/`-`, month-first for `/`) and flagged in
  `diagnostics.notes` when ambiguous.
- Learned spans are trained on synthetic text (Faker names/addresses in 30+ locales,
  templated sentences). Real-world precision is lower than the held-out numbers: on the
  hand-written set the model over-tags capitalised phrases as `person`/`company` and can
  mistake units, booking codes and times for entities. Person vs. company is the weakest
  distinction. See the model card.
- The model runs on 512-token windows; the kind is the average over windows, so a very long
  mixed paste is classified by its dominant flavour.
- Non-Latin scripts are tokenised as single letter runs (no word boundaries), so CJK names
  and addresses rely mostly on line-position and punctuation features.
- Nothing here is a validator: `email` means "looks like an e-mail", not "deliverable".

See [MODEL_CARD.md](./MODEL_CARD.md) for evaluation numbers and data sources.

## Training

```bash
cd packages/gpu-paste/training
uv sync
uv run python -m gpu_paste.data 20     # print 20 generated examples
uv run python -m gpu_paste.train       # 160K synthetic examples, 8 epochs, int6 QAT (~8 min, 2 threads)
uv run python -m gpu_paste.export      # write ../model/{manifest.json,weights.txt,fixtures.json}
uv run python -m gpu_paste.evaluate    # held-out confusion matrix + per-span F1, unfamiliar set
uv run pytest                          # includes running shader.wgsl on lavapipe vs. the fixtures
```
