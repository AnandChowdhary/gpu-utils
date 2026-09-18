"""Runs the runtime's canonical scan kernel (packages/runtime/src/wgsl/scan_tagger.wgsl) on a
real WebGPU adapter (Mesa lavapipe on CI) through gpu_utils_training.kernels, exactly like
runtime/gpu.ts runScanTagger, and checks both heads against model/fixtures.json and the
Python reference. Skips without an adapter."""

import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights, flat_weights

from gpu_paste.features import featurize

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def _load() -> tuple[dict, np.ndarray]:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    return manifest, flat_weights((MODEL_DIR / "weights.txt").read_text(), manifest)


def test_wgsl_matches_fixtures(wgpu_device) -> None:
    manifest, weights = _load()
    cases = [c for c in json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"] if c["rows"]]
    tags, pooled = run_tagger(make_runner(manifest, wgpu_device), manifest, weights, [c["rows"] for c in cases])
    assert pooled is not None and len(tags) == len(pooled) == len(cases)
    worst = 0.0
    for c, t, p in zip(cases, tags, pooled, strict=True):
        worst = max(worst, float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max()))
        worst = max(worst, float(np.abs(p - np.asarray(c["pooled"], dtype=np.float32)).max()))
    print(f"WGSL vs fixtures over {len(cases)} sequences in one dispatch: max |Δ| = {worst:.2e}")
    assert worst < 1e-4


def test_wgsl_long_window_matches_reference(wgpu_device) -> None:
    """A full 512-token inference window (what src/index.ts dispatches) against the Python family forward."""
    manifest, weights = _load()
    text = ("Invoice from Acme Corp for $12.50, call +1 555 0100 by Friday.\n" * 40).strip()
    rows = featurize(text)[:512]
    assert len(rows) == 512
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    r, mask = collate([rows], manifest["slots"], manifest["paddingId"])
    with torch.no_grad():
        ref = model(r, mask)
    tags, pooled = run_tagger(make_runner(manifest, wgpu_device), manifest, weights, [rows])
    assert pooled is not None
    assert np.abs(tags[0] - ref["tags"][0].numpy()).max() < 1e-4
    assert np.abs(pooled[0] - ref["pooled"][0].numpy()).max() < 1e-4
