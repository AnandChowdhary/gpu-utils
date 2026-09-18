import { describe, expect, it } from "vitest";
import { argmax, viterbi } from "../src/decode.ts";
import { decodeInt6, type TensorEntry } from "../src/weights.ts";

describe("argmax", () => {
  it("picks the max per row", () => {
    expect(Array.from(argmax(new Float32Array([0, 1, 2, 5, 3, 1]), 2, 3))).toEqual([2, 0]);
  });
});

describe("viterbi", () => {
  it("forbids O -> I transitions", () => {
    // labels: 0 = O, 1 = B, 2 = I. Emissions favour I at t=1 but O at t=0.
    const k = 3;
    const em = new Float32Array([5, 0, 0, 0, 1, 4, 0, 0, 4]);
    const tr = new Float32Array(k * k).fill(0);
    tr[0 * k + 2] = -Infinity; // O -> I
    expect(Array.from(viterbi(em, 3, k, tr))).toEqual([0, 1, 2]);
  });
  it("handles empty input", () => {
    expect(viterbi(new Float32Array(0), 0, 2, new Float32Array(4)).length).toBe(0);
  });
});

describe("decodeInt6", () => {
  it("matches the Python encoder for [0.5, -0.25, 0, 1.0]", () => {
    // From tooling/python/tests/test_parity.py: codes 48, 24, 32, 63 with scale 1/31.
    const alphabet = (() => {
      let s = "";
      for (let c = 35; c < 127 && s.length < 64; c++) {
        const ch = String.fromCharCode(c);
        if (!"\\\"'`".includes(ch)) s += ch;
      }
      return s;
    })();
    const encoded = alphabet[48]! + alphabet[24]! + alphabet[32]! + alphabet[63]!;
    const tensors: TensorEntry[] = [{ name: "w", offset: 0, length: 4, shape: [4], scale: 1 / 31 }];
    const w = decodeInt6(encoded, tensors);
    expect(Array.from(w).map((v) => Math.round(v * 1000) / 1000)).toEqual([0.516, -0.258, 0, 1]);
  });
});
