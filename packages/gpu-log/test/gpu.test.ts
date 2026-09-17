import { describe, expect, it } from "vitest";
import { grid, weightOffsets } from "../src/gpu.ts";
import { MODEL } from "../src/model.ts";

/**
 * The WebGPU path cannot run under Node without a device; test/gpu-parity.mjs runs the real
 * shader against the CPU path when Dawn bindings are available. Here we check the pieces that
 * must agree with shader.wgsl: the offset table order and the dispatch grid.
 */
describe("gpu plumbing", () => {
  it("lays out weight offsets in the order the shader reads them", () => {
    const offs = weightOffsets(MODEL);
    expect(offs).toHaveLength(3 + 4 * 5 + 6);
    expect(offs[0]).toBe(MODEL.offsets.embed);
    expect(offs[3]).toBe(MODEL.offsets.block0_w1);
    expect(offs[19]).toBe(MODEL.offsets.block4_w1);
    expect(offs[23]).toBe(MODEL.offsets.head_h_w);
    expect(offs[28]).toBe(MODEL.offsets.head_kind_b);
  });
  it("wraps the workgroup grid at 32768 columns", () => {
    expect(grid(10)).toEqual([10, 1]);
    expect(grid(32768)).toEqual([32768, 1]);
    expect(grid(32769)).toEqual([32768, 2]);
  });
});
