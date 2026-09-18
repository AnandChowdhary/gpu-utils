import { describe, expect, it } from "vitest";
import { grid, packRows, taggerParams, tensorOffsets, unpackLogits } from "../src/batch.ts";
import {
  convTaggerEntries,
  convTaggerPasses,
  convTaggerShader,
  scanTaggerEntries,
  scanTaggerPasses,
  scanTaggerShader,
} from "../src/gpu.ts";
import type { ModelManifest } from "../src/weights.ts";

describe("packRows", () => {
  it("pads slots and tokens with the padding id", () => {
    const { rows, lengths, maxTokens } = packRows([[[1, 2, 3], [4]], [[5]]], 2, 99);
    expect(maxTokens).toBe(2);
    expect(Array.from(rows)).toEqual([1, 2, 4, 99, 5, 99, 99, 99]);
    expect(Array.from(lengths)).toEqual([2, 1]);
  });
  it("handles empty sequences", () => {
    const { rows, lengths, maxTokens } = packRows([[]], 3, 7);
    expect(maxTokens).toBe(1);
    expect(Array.from(rows)).toEqual([7, 7, 7]);
    expect(Array.from(lengths)).toEqual([0]);
  });
});

describe("tensorOffsets / taggerParams / grid", () => {
  const manifest = {
    format: 1,
    name: "x",
    labels: [],
    parameters: 0,
    tensors: [
      { name: "a", offset: 0, length: 4, shape: [4], scale: 1 },
      { name: "b", offset: 4, length: 2, shape: [2], scale: 1 },
    ],
  } satisfies ModelManifest;
  it("looks offsets up by name in order", () => {
    expect(Array.from(tensorOffsets(manifest, ["b", "a"]))).toEqual([4, 0]);
    expect(() => tensorOffsets(manifest, ["c"])).toThrow(/unknown tensor/);
  });
  it("builds the 16-word uniform block in kernel order", () => {
    const p = taggerParams({
      batch: 2,
      maxTokens: 5,
      slots: 3,
      padding: 9,
      embed: 8,
      hidden: 8,
      head: 16,
      tags: 4,
      pooled: 0,
      layers: 1,
      taps: 5,
    });
    expect(p.length).toBe(16);
    expect(Array.from(p.subarray(0, 11))).toEqual([2, 5, 3, 9, 8, 8, 16, 4, 0, 1, 5]);
  });
  it("wraps the dispatch grid at 32768", () => {
    expect(grid(10)).toEqual([10, 1]);
    expect(grid(70000)).toEqual([32768, 3]);
    expect(grid(0)).toEqual([1, 1]);
  });
});

describe("unpackLogits", () => {
  it("splits per sequence and strips padding", () => {
    // batch 2, maxTokens 2, tags 2, pooled 1
    const out = Float32Array.from([1, 2, 3, 4, 5, 6, 0, 0, 10, 20]);
    const r = unpackLogits(out, Uint32Array.from([2, 1]), 2, 2, 1);
    expect(r.tags.map((t) => Array.from(t))).toEqual([
      [1, 2, 3, 4],
      [5, 6],
    ]);
    expect(r.pooled?.map((p) => Array.from(p))).toEqual([[10], [20]]);
    expect(unpackLogits(out, Uint32Array.from([2, 1]), 2, 2, 0).pooled).toBeUndefined();
  });
});

describe("kernel entry points and passes", () => {
  it("lists entries per layer/block and rejects too many", () => {
    expect(scanTaggerEntries(2)).toEqual([
      "embed",
      "gates0",
      "scan0",
      "conv0",
      "gates1",
      "scan1",
      "conv1",
      "pool",
      "head",
      "pooled",
    ]);
    expect(convTaggerEntries(2)).toEqual(["embed", "block0", "block1", "head", "pool"]);
    expect(() => scanTaggerEntries(5)).toThrow();
    expect(() => convTaggerEntries(9)).toThrow();
  });
  it("every entry point exists in the shader source", () => {
    for (const e of scanTaggerEntries(4))
      expect(scanTaggerShader).toMatch(new RegExp(`fn ${e}\\(`));
    for (const e of convTaggerEntries(8))
      expect(convTaggerShader).toMatch(new RegExp(`fn ${e}\\(`));
  });
  it("dispatches per-token passes on the grid and per-sequence passes on the batch", () => {
    const scan = scanTaggerPasses(1, 100, 3);
    expect(scan.map((p) => p.entry)).toEqual([
      "embed",
      "gates0",
      "scan0",
      "conv0",
      "pool",
      "head",
      "pooled",
    ]);
    expect(scan[0]!.workgroups).toEqual([100, 1]);
    expect(scan[2]!.workgroups).toEqual([3]);
    const conv = convTaggerPasses(3, 40000, 2);
    expect(conv.map((p) => p.entry)).toEqual([
      "embed",
      "block0",
      "block1",
      "block2",
      "head",
      "pool",
    ]);
    expect(conv[0]!.workgroups).toEqual([32768, 2]);
    expect(conv.at(-1)!.workgroups).toEqual([2]);
  });
});
