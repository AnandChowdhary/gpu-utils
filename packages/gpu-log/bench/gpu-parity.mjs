/**
 * Builds a self-contained browser page that runs the WGSL forward pass against the CPU
 * reference on model/fixtures.json and reports the max |difference| per fixture (must be < 1e-4).
 *
 *   node packages/gpu-log/bench/gpu-parity.mjs && open packages/gpu-log/bench/dist/gpu-parity.html
 *
 * Node has no WebGPU device, so this check runs in Chrome/Edge (or Firefox/Safari with WebGPU).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = resolve(here, "..");
const entry = `
import { tokenize } from "@gpu-utils/runtime";
import fixtures from "../model/fixtures.json" with { type: "json" };
import { forwardCpu } from "../src/cpu.ts";
import { FEATURE_COUNT, writeTokenFeatures } from "../src/features.ts";
import { forwardGpu } from "../src/gpu.ts";
import { MODEL } from "../src/model.ts";

const log = (s) => { document.body.appendChild(Object.assign(document.createElement("pre"), { textContent: s })); };
(async () => {
  if (!navigator.gpu) { log("WebGPU unavailable"); return; }
  const cases = fixtures.filter((c) => c.text.length > 0);
  const tokens = cases.map((c) => tokenize(c.text));
  const n = tokens.reduce((s, t) => s + t.length, 0);
  const features = new Uint32Array(n * FEATURE_COUNT);
  const lineId = new Uint32Array(n);
  const offsets = [0];
  let p = 0;
  for (const [li, toks] of tokens.entries()) {
    for (let i = 0; i < toks.length; i++, p++) { writeTokenFeatures(toks, i, features, p * FEATURE_COUNT); lineId[p] = li; }
    offsets.push(p);
  }
  const batch = { n, features, lineId, offsets };
  const t0 = performance.now();
  const gpu = await forwardGpu(MODEL, batch);
  const t1 = performance.now();
  const cpu = forwardCpu(MODEL, batch);
  const t2 = performance.now();
  const W = MODEL.tags + MODEL.kinds;
  let worst = 0;
  for (const [li, c] of cases.entries()) {
    let d = 0;
    for (let q = offsets[li] * W; q < offsets[li + 1] * W; q++) d = Math.max(d, Math.abs(gpu[q] - cpu[q]));
    worst = Math.max(worst, d);
    log((d < 1e-4 ? "ok   " : "FAIL ") + d.toExponential(2) + "  " + JSON.stringify(c.text.slice(0, 60)));
  }
  log("max |gpu - cpu| = " + worst.toExponential(2) + " over " + n + " tokens; gpu " + (t1 - t0).toFixed(1) + " ms (cold), cpu " + (t2 - t1).toFixed(1) + " ms");
  const t3 = performance.now();
  await forwardGpu(MODEL, batch);
  log("gpu warm: " + (performance.now() - t3).toFixed(1) + " ms");
})();
`;
mkdirSync(resolve(here, "dist"), { recursive: true });
writeFileSync(resolve(here, "dist/entry.ts"), entry);
const result = await build({
  stdin: { contents: entry, resolveDir: resolve(pkg, "bench"), loader: "ts" },
  bundle: true,
  write: false,
  format: "esm",
  platform: "browser",
  target: "es2022",
  loader: { ".txt": "text", ".wgsl": "text" },
  absWorkingDir: pkg,
});
const js = result.outputFiles[0].text;
writeFileSync(
  resolve(here, "dist/gpu-parity.html"),
  `<!doctype html><meta charset="utf-8"><title>gpu-log WGSL parity</title><body style="font:14px monospace"><script type="module">${js}</script></body>`,
);
console.log(
  `wrote ${resolve(here, "dist/gpu-parity.html")} (${(js.length / 1024).toFixed(0)} KB); open it in a WebGPU browser`,
);
