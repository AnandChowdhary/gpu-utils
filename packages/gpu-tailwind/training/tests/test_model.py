import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights

from gpu_tailwind.batches import encode, loss, make_batch
from gpu_tailwind.features import WIDTH, featurize
from gpu_tailwind.labels import LABELS
from gpu_tailwind.model import TAGS, build

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_forward_and_loss() -> None:
    model = build()
    rows, mask = collate([featurize("bold red text"), featurize("hi")], WIDTH, model.padding_id)
    out = model(rows, mask)
    assert out["tags"].shape == (2, 5, TAGS) and out["pooled"] is None
    assert TAGS == len(LABELS) + 1
    data = encode([{"text": "bold red text", "labels": ["B-VAL", "O", "B-VAL", "O", "B-PROP"], "boundary": [1, 0, 0, 0, 0]}])
    assert torch.isfinite(loss(model, make_batch(data, model.padding_id)))


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the Python family forward reproduce model/fixtures.json."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    model.eval()
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
