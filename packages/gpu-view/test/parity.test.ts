import { describe, it } from "vitest";

/**
 * Parity fixtures are written by `pnpm export` (training/gpu_view/export.py) into
 * model/fixtures.json: { text, features, logits } per case. The CPU reference path
 * must reproduce the PyTorch logits within tolerance; the WebGPU path is checked
 * against the CPU path in the browser test suite.
 */
describe("gpu-view parity", () => {
  it.todo("matches PyTorch fixtures on the CPU reference path");
});
