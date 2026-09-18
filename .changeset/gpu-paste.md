---
"gpu-paste": minor
---

Initial release of gpu-paste: understand pasted text on-device. Deterministic detectors
for JSON, CSV/TSV, HTML, URL, email, phone, datetime, color, UUID, JWT, IP, path, money and
number; a 66K-parameter int6 bidirectional affine-scan model classifies address / contact /
prose / list / code / markdown and tags person, company, address, date, money and phone spans
inside mixed text. CPU reference path plus WGSL kernels, zero runtime dependencies.
