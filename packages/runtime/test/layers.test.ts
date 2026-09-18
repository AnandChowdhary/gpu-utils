import { describe, expect, it } from "vitest";
import {
  affineScan,
  biScan,
  concatRows,
  dense,
  depthwiseConv,
  dilatedResidualBlock,
  meanPool,
  relu,
  sigmoid,
  sparseEmbed,
} from "../src/layers.ts";

const close = (a: ArrayLike<number>, b: number[], tol = 1e-6) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(tol);
};

describe("layers", () => {
  it("sparseEmbed sums rows and skips the padding id", () => {
    const table = Float32Array.from({ length: 12 }, (_, i) => i); // 4 rows x 3
    const out = sparseEmbed([[0, 4, 4], [1, 2], [4]], table, 3, 4);
    close(out, [0, 1, 2, 9, 11, 13, 0, 0, 0]);
  });

  it("dense applies x @ W + b with W [in, out]", () => {
    const w = Float32Array.from([1, 2, 3, 4, 5, 6]); // [2, 3]
    const out = dense(Float32Array.from([1, 1, 0, 2]), 2, 2, w, Float32Array.from([10, 20, 30]), 3);
    close(out, [15, 27, 39, 18, 30, 42]);
  });

  it("affineScan runs the recurrence forward and backward", () => {
    const a = Float32Array.from([0.5, 0.5, 0.5]);
    const b = Float32Array.from([1, 1, 1]);
    close(affineScan(a, b, 3, 1), [1, 1.5, 1.75]);
    close(affineScan(a, b, 3, 1, true), [1.75, 1.5, 1]);
    // channels are independent
    const two = affineScan(Float32Array.from([1, 0, 1, 0]), Float32Array.from([1, 2, 1, 3]), 2, 2);
    close(two, [1, 2, 2, 3]);
  });

  it("depthwiseConv is zero padded with the centre tap in the middle", () => {
    const w = Float32Array.from([1, 10, 100]); // taps=3, c=1: x[t-1] + 10 x[t] + 100 x[t+1]
    const out = depthwiseConv(Float32Array.from([1, 2, 3]), 3, 1, w, Float32Array.from([0.5]), 3);
    close(out, [210.5, 321.5, 32.5]);
  });

  it("dilatedResidualBlock: x + relu(conv3_d(x)) @ w2 + b2", () => {
    // h = 1, dilation 2, w1 = [tap-1: 1, tap0: 0, tap+1: -1], w2 = 2, b1 = 0, b2 = 0.5
    const x = Float32Array.from([1, 2, 3, 4, 5]);
    const out = dilatedResidualBlock(
      x,
      5,
      1,
      Float32Array.from([1, 0, -1]),
      Float32Array.from([0]),
      Float32Array.from([2]),
      Float32Array.from([0.5]),
      2,
    );
    // conv: t0: -x2 = -3 -> relu 0; t1: -x3 = -4 -> 0; t2: x0 - x4 = -4 -> 0; t3: x1 = 2; t4: x2 = 3
    close(out, [1.5, 2.5, 3.5, 8.5, 11.5]);
  });

  it("biScan gates and combines both directions", () => {
    // dIn = 1, hidden = 1; forward: a = sigmoid(0) = 0.5, u = tanh(x); backward: a = sigmoid(10) ~ 1 -> h ~ 0
    const w = {
      fWa: Float32Array.from([0]),
      fBa: Float32Array.from([0]),
      fWu: Float32Array.from([1]),
      fBu: Float32Array.from([0]),
      bWa: Float32Array.from([0]),
      bBa: Float32Array.from([10]),
      bWu: Float32Array.from([1]),
      bBu: Float32Array.from([0]),
    };
    const x = Float32Array.from([1, -1]);
    const out = biScan(x, 2, 1, 1, w);
    const u0 = Math.tanh(1);
    const u1 = Math.tanh(-1);
    const h0 = 0.5 * u0;
    const h1 = 0.5 * h0 + 0.5 * u1;
    const ab = sigmoid(10);
    const hb1 = (1 - ab) * u1;
    const hb0 = ab * hb1 + (1 - ab) * u0;
    close(out, [h0, hb0, h1, hb1]);
  });

  it("meanPool, relu, concatRows", () => {
    close(meanPool(Float32Array.from([1, 2, 3, 4]), 2, 2), [2, 3]);
    close(meanPool(new Float32Array(0), 0, 2), [0, 0]);
    close(relu(Float32Array.from([-1, 0, 2])), [0, 0, 2]);
    close(
      concatRows(Float32Array.from([1, 2, 3, 4]), 2, Float32Array.from([9]), 1, 2),
      [1, 2, 9, 3, 4, 9],
    );
    close(concatRows(Float32Array.from([1, 2]), 1, Float32Array.from([7, 8]), 1, 2), [1, 7, 2, 8]);
  });
});
