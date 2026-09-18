import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights

from __SNAKE__.data import batches, generate, loss
from __SNAKE__.features import LABELS, SLOTS, featurize
from __SNAKE__.model import build

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_forward_and_loss() -> None:
    model = build()
    rows, mask = collate([featurize("call Ada"), featurize("hi")], SLOTS, model.padding_id)
    out = model(rows, mask)
    assert out["tags"].shape == (2, 3, len(LABELS)) and out["pooled"] is None
    batch = next(batches(generate(8, seed=0), 4, np.random.default_rng(0), model.padding_id))
    assert torch.isfinite(loss(model, batch))


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the Python family forward reproduce model/fixtures.json."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
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
            got = model(rows, mask)["tags"][0].numpy()
        assert np.abs(got - np.asarray(case["logits"], dtype=np.float32)).max() < 1e-4
