import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights

from gpu_paste.data import LABELS, LEARNED_KINDS, generate
from gpu_paste.dataset import Batches, encode
from gpu_paste.features import FEATURE_COUNT, featurize
from gpu_paste.model import build
from gpu_paste.train import loss

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_family_shapes_and_budget() -> None:
    model = build()
    n = model.parameter_count()
    assert 40_000 <= n <= 80_000, n
    rows, mask = collate([featurize("call Ada"), featurize("hi")], FEATURE_COUNT, model.padding_id)
    out = model(rows, mask)
    assert out["tags"].shape == (2, 3, len(LABELS))
    assert out["pooled"] is not None and out["pooled"].shape == (2, len(LEARNED_KINDS))


def test_loss_is_finite_with_masked_kinds() -> None:
    model = build()
    encoded = [encode(e) for e in generate(12, seed=3, offline=True)]
    batch = next(iter(Batches(encoded, 6, np.random.default_rng(0), model.padding_id)))
    assert torch.isfinite(loss(model, batch))
    rows, mask, labels, _kinds, idx = batch
    masked = (rows, mask, labels, torch.full((len(idx),), -1, dtype=torch.long), idx)
    assert torch.isfinite(loss(model, masked))


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the Python family forward reproduce model/fixtures.json."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    assert manifest["family"] == "scan" and manifest["slots"] == FEATURE_COUNT
    assert manifest["labels"] == LABELS and manifest["kinds"] == LEARNED_KINDS
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    cases = json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"]
    assert len(cases) >= 20
    for case in cases:
        assert featurize(case["input"]) == case["rows"]
        if not case["rows"]:
            continue
        rows, mask = collate([case["rows"]], manifest["slots"], manifest["paddingId"])
        with torch.no_grad():
            out = model(rows, mask)
        assert np.abs(out["tags"][0].numpy() - np.asarray(case["logits"], dtype=np.float32)).max() < 1e-4
        assert np.abs(out["pooled"][0].numpy() - np.asarray(case["pooled"], dtype=np.float32)).max() < 1e-4
