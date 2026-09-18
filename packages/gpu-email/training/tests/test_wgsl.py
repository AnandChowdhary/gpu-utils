"""Runs the canonical conv kernel (packages/runtime/src/wgsl/conv_tagger.wgsl) on a real
WebGPU adapter (Mesa lavapipe on CI) exactly like src/gpu.ts does, against
model/fixtures.json and against the Python reference on a multi-thousand-token email
(dilation 32 across many workgroups, more than one 32768-column grid row is not needed).
Skips without an adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from gpu_email.features import featurize
from gpu_utils_training.batch import collate
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights, flat_weights

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def _load() -> tuple[dict, np.ndarray]:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    return manifest, flat_weights((MODEL_DIR / "weights.txt").read_text(), manifest)


def test_wgsl_matches_fixtures(wgpu_device) -> None:
    manifest, weights = _load()
    cases = [
        c
        for c in json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"]
        if c["rows"]
    ]
    assert len(cases) >= 20
    tags, _ = run_tagger(
        make_runner(manifest, wgpu_device),
        manifest,
        weights,
        [c["rows"] for c in cases],
    )
    worst = max(
        float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max())
        for c, t in zip(cases, tags, strict=True)
    )
    print(
        f"WGSL vs fixtures over {len(cases)} sequences in one dispatch: max |Δ| = {worst:.2e}"
    )
    assert worst < 1e-4


def test_wgsl_long_input(wgpu_device) -> None:
    """A >512-token input (the `auto` backend's GPU threshold) matches the Python reference."""
    manifest, weights = _load()
    text = (
        ("Thanks for the update, that works for me.\n" * 60)
        + "\nOn Mon, Jan 5, 2024 at 3:14 PM Bob <bob@example.com> wrote:\n"
        + ("> quoted line\n" * 40)
    )
    rows = featurize(text)
    assert len(rows) > 512
    model = from_config(manifest)
    model.load_tensors(
        decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest)
    )
    model.eval()
    with torch.no_grad():
        r, m = collate([rows], manifest["slots"], manifest["paddingId"])
        expected = model(r, m)["tags"][0].numpy()
    (got,), _ = run_tagger(
        make_runner(manifest, wgpu_device), manifest, weights, [rows]
    )
    assert got.shape == expected.shape
    assert float(np.abs(got - expected).max()) < 1e-4
