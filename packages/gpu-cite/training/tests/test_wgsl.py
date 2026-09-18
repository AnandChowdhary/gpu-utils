"""Runs the canonical scan_tagger.wgsl kernel (packages/runtime/src/wgsl) on a real WebGPU
adapter (Mesa lavapipe on CI) against model/fixtures.json, exactly as src/gpu.ts does
through runScanTagger. Skips without an adapter."""

import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.features import tokenize
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.quant import decode_weights, flat_weights

from gpu_cite.features import featurize_tokens
from gpu_cite.model import build

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def _model() -> tuple[dict, str]:
    return json.loads((MODEL_DIR / "manifest.json").read_text()), (MODEL_DIR / "weights.txt").read_text()


def test_wgsl_matches_fixtures_in_one_dispatch(wgpu_device) -> None:
    manifest, encoded = _model()
    cases = [c for c in json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"] if c["rows"]]
    tags, pooled = run_tagger(make_runner(manifest, wgpu_device), manifest, flat_weights(encoded, manifest), [c["rows"] for c in cases])
    assert pooled is not None
    worst = 0.0
    for c, t, p in zip(cases, tags, pooled, strict=True):
        worst = max(worst, float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max()))
        worst = max(worst, float(np.abs(p - np.asarray(c["pooled"], dtype=np.float32)).max()))
    print(f"WGSL vs fixtures over {len(cases)} references in one dispatch: max |Δ| = {worst:.2e}")
    assert worst < 1e-4


def test_wgsl_long_reference_matches_torch(wgpu_device) -> None:
    """A 300+ token reference (longer than anything in the fixtures) against the family forward."""
    manifest, encoded = _model()
    text = ("Smith, J., Doe, A. B., & Roe, C. (2019). A study of things. Journal of Stuff, 12(3), 45–67. " * 6).strip()
    rows = featurize_tokens(text, tokenize(text))
    assert len(rows) > 256
    model = build(hidden=manifest["hidden"], head=manifest["head"])
    model.load_tensors(decode_weights(encoded, manifest))
    with torch.no_grad():
        ref = model(*collate([rows], manifest["slots"], manifest["paddingId"]))
    tags, pooled = run_tagger(make_runner(manifest, wgpu_device), manifest, flat_weights(encoded, manifest), [rows])
    assert pooled is not None
    assert np.abs(tags[0] - ref["tags"][0].numpy()).max() < 1e-4
    assert np.abs(pooled[0] - ref["pooled"][0].numpy()).max() < 1e-4
