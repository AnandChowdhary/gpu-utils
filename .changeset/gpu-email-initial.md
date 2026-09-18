---
"gpu-email": minor
---

Initial release of gpu-email: splits plain-text emails into reply, attribution, quote, signature, disclaimer, forward-header, greeting and closing segments, extracts the new reply text, and pulls name, title, company, phone, email, URL and address out of the author's signature. Dilated-CNN two-head tagger (~155K params, int6) with exact rules for quote prefixes, the "-- " delimiter and email/URL regexes; WebGPU forward pass with a CPU fallback.
