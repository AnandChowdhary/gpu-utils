"""Runs the canonical scan-family WGSL kernel (packages/runtime/src/wgsl/scan_tagger.wgsl)
on a real WebGPU adapter (Mesa lavapipe on CI) against model/fixtures.json, i.e. against
the PyTorch/CPU reference. Skips without an adapter."""

import json
from pathlib import Path

import numpy as np
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.quant import flat_weights

from gpu_tailwind.features import featurize

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def _load():
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    weights = flat_weights((MODEL_DIR / "weights.txt").read_text(), manifest)
    return manifest, weights


def test_wgsl_matches_fixtures(wgpu_device) -> None:
    manifest, weights = _load()
    cases = [c for c in json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"] if c["rows"]]
    tags, _ = run_tagger(make_runner(manifest, wgpu_device), manifest, weights, [c["rows"] for c in cases])
    worst = max(float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max()) for c, t in zip(cases, tags, strict=True))
    print(f"WGSL vs fixtures over {len(cases)} sequences in one dispatch: max |Δ| = {worst:.2e}")
    assert worst < 1e-4


def test_wgsl_long_input(wgpu_device) -> None:
    """A 300+ token phrase in the same batch as a short one: the sequential scan pass and
    padding handling must match the Python reference."""
    import torch
    from gpu_utils_training.batch import collate
    from gpu_utils_training.models import from_config
    from gpu_utils_training.quant import decode_weights

    manifest, weights = _load()
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    model.eval()
    long = featurize(("card with rounded corners, subtle shadow, blue on hover, hidden on mobile, " * 12).strip())
    short = featurize("bold red text")
    assert len(long) > 256
    tags, _ = run_tagger(make_runner(manifest, wgpu_device), manifest, weights, [long, short])
    for rows, got in zip([long, short], tags, strict=True):
        r, m = collate([rows], manifest["slots"], manifest["paddingId"])
        with torch.no_grad():
            ref = model(r, m)["tags"][0].numpy()
        assert np.abs(got - ref).max() < 1e-4
