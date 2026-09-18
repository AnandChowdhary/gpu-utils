/**
 * CPU reference layers. Mirrors tooling/python/gpu_utils_training/layers.py op for op.
 *
 * Every function works on ONE sequence of `n` tokens stored row-major in a Float32Array
 * (`x[t * c + i]`), accumulates in doubles and rounds to float32 on store, like torch.
 * Weight layouts: dense `[in, out]` (`w[i * out + o]`), depthwise conv `[taps, c]`,
 * dilated block `w1[tap][in][out]` = `w1[(tap * h + i) * h + o]`.
 */

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** In-place ReLU. */
export function relu(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) if (x[i]! < 0) x[i] = 0;
  return x;
}

/** Sum of embedding rows per token; ids equal to `paddingId` contribute nothing. */
export function sparseEmbed(
  rows: ArrayLike<ArrayLike<number>>,
  table: Float32Array,
  dim: number,
  paddingId: number,
): Float32Array {
  const n = rows.length;
  const out = new Float32Array(n * dim);
  for (let t = 0; t < n; t++) {
    const row = rows[t]!;
    for (let s = 0; s < row.length; s++) {
      const id = row[s]!;
      if (id === paddingId) continue;
      for (let c = 0; c < dim; c++) out[t * dim + c]! += table[id * dim + c]!;
    }
  }
  return out;
}

/** `y = x @ w + b` for `n` rows: x `[n, dIn]`, w `[dIn, dOut]`, b `[dOut]`. */
export function dense(
  x: Float32Array,
  n: number,
  dIn: number,
  w: Float32Array,
  b: Float32Array,
  dOut: number,
): Float32Array {
  const out = new Float32Array(n * dOut);
  const acc = new Float64Array(dOut);
  for (let t = 0; t < n; t++) {
    for (let o = 0; o < dOut; o++) acc[o] = b[o]!;
    for (let i = 0; i < dIn; i++) {
      const v = x[t * dIn + i]!;
      if (v === 0) continue;
      for (let o = 0; o < dOut; o++) acc[o]! += v * w[i * dOut + o]!;
    }
    for (let o = 0; o < dOut; o++) out[t * dOut + o] = acc[o]!;
  }
  return out;
}

/** Inclusive scan `h_t = a_t * h_{t-1} + b_t` per channel; `reverse` runs from the last token. */
export function affineScan(
  a: Float32Array,
  b: Float32Array,
  n: number,
  c: number,
  reverse = false,
): Float32Array {
  const out = new Float32Array(n * c);
  for (let ch = 0; ch < c; ch++) {
    let state = 0;
    for (let step = 0; step < n; step++) {
      const t = reverse ? n - 1 - step : step;
      state = a[t * c + ch]! * state + b[t * c + ch]!;
      out[t * c + ch] = state;
    }
  }
  return out;
}

export interface BiScanWeights {
  fWa: Float32Array;
  fBa: Float32Array;
  fWu: Float32Array;
  fBu: Float32Array;
  bWa: Float32Array;
  bBa: Float32Array;
  bWu: Float32Array;
  bBu: Float32Array;
}

/**
 * Gated bidirectional affine scan: x `[n, dIn]` → `[n, 2 * hidden]` (forward ‖ backward).
 * Per direction `a = sigmoid(x Wa + ba)`, `u = tanh(x Wu + bu)`, `h = a * h_prev + (1 - a) * u`.
 */
export function biScan(
  x: Float32Array,
  n: number,
  dIn: number,
  hidden: number,
  w: BiScanWeights,
): Float32Array {
  const out = new Float32Array(n * 2 * hidden);
  for (const reverse of [false, true]) {
    const wa = reverse ? w.bWa : w.fWa;
    const ba = reverse ? w.bBa : w.fBa;
    const wu = reverse ? w.bWu : w.fWu;
    const bu = reverse ? w.bBu : w.fBu;
    const a = dense(x, n, dIn, wa, ba, hidden);
    const u = dense(x, n, dIn, wu, bu, hidden);
    for (let i = 0; i < a.length; i++) {
      const g = sigmoid(a[i]!);
      a[i] = g;
      u[i] = (1 - g) * Math.tanh(u[i]!);
    }
    const h = affineScan(a, u, n, hidden, reverse);
    const off = reverse ? hidden : 0;
    for (let t = 0; t < n; t++)
      for (let ch = 0; ch < hidden; ch++) out[t * 2 * hidden + off + ch] = h[t * hidden + ch]!;
  }
  return out;
}

/** Per-channel convolution over time, zero padded, centre tap at `taps >> 1`; w `[taps, c]`. */
export function depthwiseConv(
  x: Float32Array,
  n: number,
  c: number,
  w: Float32Array,
  b: Float32Array,
  taps: number,
): Float32Array {
  const out = new Float32Array(n * c);
  const half = taps >> 1;
  for (let t = 0; t < n; t++) {
    for (let ch = 0; ch < c; ch++) {
      let acc = b[ch]!;
      for (let k = 0; k < taps; k++) {
        const src = t + k - half;
        if (src >= 0 && src < n) acc += w[k * c + ch]! * x[src * c + ch]!;
      }
      out[t * c + ch] = acc;
    }
  }
  return out;
}

/** `y = x + relu(conv3_dilated(x)) @ w2 + b2`; w1 `[3, h, h]`, w2 `[h, h]`. */
export function dilatedResidualBlock(
  x: Float32Array,
  n: number,
  h: number,
  w1: Float32Array,
  b1: Float32Array,
  w2: Float32Array,
  b2: Float32Array,
  dilation: number,
): Float32Array {
  const hid = new Float32Array(n * h);
  const acc = new Float64Array(h);
  for (let t = 0; t < n; t++) {
    for (let o = 0; o < h; o++) acc[o] = b1[o]!;
    for (let tap = 0; tap < 3; tap++) {
      const q = t + (tap - 1) * dilation;
      if (q < 0 || q >= n) continue;
      for (let i = 0; i < h; i++) {
        const v = x[q * h + i]!;
        if (v === 0) continue;
        const wo = (tap * h + i) * h;
        for (let o = 0; o < h; o++) acc[o]! += v * w1[wo + o]!;
      }
    }
    for (let o = 0; o < h; o++) hid[t * h + o] = Math.max(acc[o]!, 0);
  }
  const y = dense(hid, n, h, w2, b2, h);
  for (let i = 0; i < y.length; i++) y[i] = x[i]! + y[i]!;
  return y;
}

/** Mean over tokens: `[n, c]` → `[c]` (zeros when n = 0). */
export function meanPool(x: Float32Array, n: number, c: number): Float32Array {
  const out = new Float32Array(c);
  if (n === 0) return out;
  const acc = new Float64Array(c);
  for (let t = 0; t < n; t++) for (let ch = 0; ch < c; ch++) acc[ch]! += x[t * c + ch]!;
  for (let ch = 0; ch < c; ch++) out[ch] = acc[ch]! / n;
  return out;
}

/** `[n, a] ‖ [n, b]` → `[n, a + b]`; `right` may be a single `[b]` vector broadcast to every row. */
export function concatRows(
  left: Float32Array,
  a: number,
  right: Float32Array,
  b: number,
  n: number,
): Float32Array {
  const out = new Float32Array(n * (a + b));
  const broadcast = right.length === b;
  for (let t = 0; t < n; t++) {
    out.set(left.subarray(t * a, (t + 1) * a), t * (a + b));
    const r = broadcast ? right : right.subarray(t * b, (t + 1) * b);
    out.set(r, t * (a + b) + a);
  }
  return out;
}
