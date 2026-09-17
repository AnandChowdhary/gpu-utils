import { type Model, view } from "./model.ts";

export interface Logits {
  /** [tokens, labels] span logits, row-major. */
  span: Float32Array;
  /** [kinds] logits from the mean-pooled kind head. */
  kind: Float32Array;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * Reference forward pass in plain TypeScript, mirroring training/gpu_paste/model.py:
 * summed sparse embeddings → bidirectional gated affine scans → pointwise mix →
 * mean-pooled context → BIO span head and kind head. Source of truth for shader.wgsl.
 */
export function forwardCpu(model: Model, rows: number[][]): Logits {
  const { dim: D, mix: M, kindHidden: KH } = model.manifest;
  const L = model.manifest.labels.length;
  const K = model.manifest.kinds.length;
  const n = rows.length;
  const E = view(model, "embed");
  const gates = [
    { w: view(model, "gate_f.w"), b: view(model, "gate_f.b") },
    { w: view(model, "gate_b.w"), b: view(model, "gate_b.b") },
  ];
  const Wm = view(model, "mix.w");
  const bm = view(model, "mix.b");
  const Ws = view(model, "head.w");
  const bs = view(model, "head.b");
  const Wo = view(model, "out.w");
  const bo = view(model, "out.b");
  const Wk1 = view(model, "kind1.w");
  const bk1 = view(model, "kind1.b");
  const Wk2 = view(model, "kind2.w");
  const bk2 = view(model, "kind2.b");

  // Embedding: x_t = sum_f E[id_f]
  const x = new Float32Array(n * D);
  for (let t = 0; t < n; t++) {
    for (const id of rows[t]!) {
      const base = id * D;
      for (let d = 0; d < D; d++) x[t * D + d] += E[base + d]!;
    }
  }

  // Bidirectional gated affine scans: h_t = a_t * h_{t-1} + (1 - a_t) * u_t
  const h = [new Float32Array(n * D), new Float32Array(n * D)];
  for (let dir = 0; dir < 2; dir++) {
    const { w, b } = gates[dir]!;
    const out = h[dir]!;
    const prev = new Float32Array(D);
    for (let step = 0; step < n; step++) {
      const t = dir === 0 ? step : n - 1 - step;
      for (let d = 0; d < D; d++) {
        let pa = b[d]!;
        let pu = b[D + d]!;
        for (let i = 0; i < D; i++) {
          const xi = x[t * D + i]!;
          pa += w[d * D + i]! * xi;
          pu += w[(D + d) * D + i]! * xi;
        }
        const a = sigmoid(pa);
        const v = a * prev[d]! + (1 - a) * Math.tanh(pu);
        prev[d] = v;
        out[t * D + d] = v;
      }
    }
  }

  // Pointwise mix over [x; h_forward; h_backward], then mean pool.
  const m = new Float32Array(n * M);
  const g = new Float32Array(M);
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < M; j++) {
      let acc = bm[j]!;
      const row = j * 3 * D;
      for (let i = 0; i < D; i++) {
        acc += Wm[row + i]! * x[t * D + i]! + Wm[row + D + i]! * h[0]![t * D + i]! + Wm[row + 2 * D + i]! * h[1]![t * D + i]!;
      }
      const v = acc > 0 ? acc : 0;
      m[t * M + j] = v;
      g[j] = g[j]! + v;
    }
  }
  if (n > 0) for (let j = 0; j < M; j++) g[j] = g[j]! / n;

  // Span head: s_t = relu(Ws [m_t; g] + bs); logits = Wo s_t + bo
  const span = new Float32Array(n * L);
  const s = new Float32Array(M);
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < M; j++) {
      let acc = bs[j]!;
      const row = j * 2 * M;
      for (let i = 0; i < M; i++) acc += Ws[row + i]! * m[t * M + i]! + Ws[row + M + i]! * g[i]!;
      s[j] = acc > 0 ? acc : 0;
    }
    for (let l = 0; l < L; l++) {
      let acc = bo[l]!;
      for (let i = 0; i < M; i++) acc += Wo[l * M + i]! * s[i]!;
      span[t * L + l] = acc;
    }
  }

  // Kind head on the pooled context.
  const kh = new Float32Array(KH);
  for (let j = 0; j < KH; j++) {
    let acc = bk1[j]!;
    for (let i = 0; i < M; i++) acc += Wk1[j * M + i]! * g[i]!;
    kh[j] = acc > 0 ? acc : 0;
  }
  const kind = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    let acc = bk2[k]!;
    for (let i = 0; i < KH; i++) acc += Wk2[k * KH + i]! * kh[i]!;
    kind[k] = acc;
  }
  return { span, kind };
}
