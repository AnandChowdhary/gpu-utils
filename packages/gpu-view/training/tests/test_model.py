import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights

from gpu_view import data, features
from gpu_view.generate import ROLES, dataset
from gpu_view.model import TAGS, build

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_forward_and_loss() -> None:
    model = build()
    assert model.parameter_count() < 45000
    enc = data.encode(dataset("train", 8, seed=0))
    batch = data.batches(enc, 4, np.random.default_rng(0))[0]
    rows, mask, _roles, _bounds = batch
    out = model(rows, mask)
    assert out["tags"].shape[-1] == TAGS == len(ROLES) + 1
    assert torch.isfinite(data.loss(model, batch))


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the Python family forward reproduce model/fixtures.json."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    cases = json.loads((MODEL_DIR / "fixtures.json").read_text())["cases"]
    assert len(cases) >= 20
    assert manifest["labels"] == ROLES and manifest["tags"] == len(ROLES) + 1
    for case in cases:
        _toks, rows = features.featurize(case["input"]["text"], case["input"]["schema"])
        assert rows == case["rows"]
        if not rows:
            continue
        r, mask = collate([rows], manifest["slots"], manifest["paddingId"])
        with torch.no_grad():
            got = model(r, mask)["tags"][0].numpy()
        assert np.abs(got - np.asarray(case["logits"], dtype=np.float32)).max() < 1e-4
