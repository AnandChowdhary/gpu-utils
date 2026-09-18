# gpu-paste

## 0.1.0

### Minor Changes

- [#1](https://github.com/AnandChowdhary/gpu-utils/pull/1) [`e600415`](https://github.com/AnandChowdhary/gpu-utils/commit/e6004157b31d24a3b67f3d6bf42d8c54b28ab84c) Thanks [@AnandChowdhary](https://github.com/AnandChowdhary)! - Initial release of gpu-paste: understand pasted text on-device. Deterministic detectors
  for JSON, CSV/TSV, HTML, URL, email, phone, datetime, color, UUID, JWT, IP, path, money and
  number; a 66K-parameter int6 bidirectional affine-scan model classifies address / contact /
  prose / list / code / markdown and tags person, company, address, date, money and phone spans
  inside mixed text. CPU reference path plus WGSL kernels, zero runtime dependencies.
