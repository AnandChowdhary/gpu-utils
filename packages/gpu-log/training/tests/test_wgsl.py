"""Runs the canonical WGSL kernel (packages/runtime/src/wgsl) on a real WebGPU adapter
(Mesa lavapipe on CI) against model/fixtures.json. Skips without an adapter."""

import json
from pathlib import Path

import numpy as np
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.quant import flat_weights

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_wgsl_matches_fixtures(wgpu_device) -> None:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    weights = flat_weights((MODEL_DIR / "weights.txt").read_text(), manifest)
    cases = [c for c in json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"] if c["rows"]]
    tags, _ = run_tagger(make_runner(manifest, wgpu_device), manifest, weights, [c["rows"] for c in cases])
    worst = max(float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max()) for c, t in zip(cases, tags, strict=True))
    print(f"WGSL vs fixtures over {len(cases)} sequences in one dispatch: max |Δ| = {worst:.2e}")
    assert worst < 1e-4
