"""TS CPU == Python == WGSL: the committed family fixtures (also read by models.test.ts)
are pushed through the canonical kernels via WgslRunner, and a larger fresh random model
is compared against torch directly. Skips without a WebGPU adapter."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gpu_utils_training.batch import collate
from gpu_utils_training.fixtures import generate
from gpu_utils_training.kernels import make_runner, run_tagger
from gpu_utils_training.models import ConvTagger, ScanTagger
from gpu_utils_training.qat import set_quant
from gpu_utils_training.quant import decode_weights, flat_weights
from gpu_utils_training.quant import export as quant_export

FIXTURES = Path(__file__).resolve().parents[3] / "packages/runtime/test/fixtures"
TOL = 1e-4


def test_committed_fixtures_are_current() -> None:
    fresh = generate()
    for name, data in fresh.items():
        assert json.loads((FIXTURES / name).read_text()) == json.loads(json.dumps(data)), f"{name} is stale: run `uv run python -m gpu_utils_training.fixtures`"


@pytest.mark.parametrize("family", ["scan", "conv"])
def test_wgsl_matches_fixtures(wgpu_device, family: str) -> None:
    data = json.loads((FIXTURES / f"{family}_tagger.json").read_text())
    manifest = data["manifest"]
    weights = flat_weights(data["weights"], manifest)
    runner = make_runner(manifest, wgpu_device)
    cases = [c for c in data["cases"] if c["rows"]]
    tags, pooled = run_tagger(runner, manifest, weights, [c["rows"] for c in cases])
    worst = 0.0
    for c, t in zip(cases, tags, strict=True):
        worst = max(worst, float(np.abs(t - np.asarray(c["logits"], dtype=np.float32)).max()))
    if pooled is not None:
        for c, p in zip(cases, pooled, strict=True):
            worst = max(worst, float(np.abs(p - np.asarray(c["pooled"], dtype=np.float32)).max()))
    print(f"{family}: WGSL vs fixtures over {len(cases)} sequences in one dispatch, max |Δ| = {worst:.2e}")
    assert worst < TOL


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(lambda: ScanTagger(300, 32, 9, pooled_out=4, scan_layers=2), id="scan-h32-l2"),
        pytest.param(lambda: ScanTagger(300, 24, 6), id="scan-h24-nopool"),
        pytest.param(lambda: ConvTagger(300, 32, 64, 5, [1, 2, 4, 8, 16], 23, pooled_out=3), id="conv-h64-b5"),
        pytest.param(lambda: ConvTagger(300, 48, 48, 6, [1, 2, 4, 8, 16, 32], 15), id="conv-h48-b6-nopool"),
    ],
)
def test_wgsl_matches_torch_random_weights(wgpu_device, tmp_path: Path, model) -> None:
    torch.manual_seed(0)
    m = model()
    rng = np.random.default_rng(5)
    slots = 4
    batch = [[[int(v) for v in rng.integers(0, 300, size=rng.integers(1, slots + 1))] for _ in range(n)] for n in [1, 7, 40, 300, 13]]
    manifest = quant_export(m.tensors(), tmp_path, {"name": "t", **m.config(), "slots": slots, "labels": []})
    encoded = (tmp_path / "weights.txt").read_text()
    m.load_tensors(decode_weights(encoded, manifest))
    set_quant(m, False)
    runner = make_runner(manifest, wgpu_device)
    tags, pooled = run_tagger(runner, manifest, flat_weights(encoded, manifest), batch)
    rows, mask = collate(batch, slots, m.padding_id)
    with torch.no_grad():
        ref = m(rows, mask)
    worst = 0.0
    for i, seq in enumerate(batch):
        worst = max(worst, float(np.abs(tags[i] - ref["tags"][i, : len(seq)].numpy()).max()))
        if pooled is not None:
            worst = max(worst, float(np.abs(pooled[i] - ref["pooled"][i].numpy()).max()))
    print(f"{m.family}: WGSL vs torch on {len(batch)} sequences (up to 300 tokens), max |Δ| = {worst:.2e}")
    assert worst < TOL
