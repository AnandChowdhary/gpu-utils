import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.models import from_config
from gpu_utils_training.quant import decode_weights

from gpu_email.data import encode_example, generate
from gpu_email.features import BIO_LABELS, LINE_KINDS, NUM_SLOTS, featurize
from gpu_email.model import DILATIONS, build
from gpu_email.train import collate_examples, loss

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def test_forward_and_loss() -> None:
    model = build()
    assert model.config()["dilations"] == DILATIONS
    rows, mask = collate(
        [featurize("Hi Bob,\nthanks\n"), featurize("ok")], NUM_SLOTS, model.padding_id
    )
    out = model(rows, mask)
    assert (
        out["tags"].shape == (2, rows.shape[1], len(LINE_KINDS) + len(BIO_LABELS))
        and out["pooled"] is None
    )
    batch = collate_examples(
        [encode_example(ex) for ex in generate(5, 4)], model.padding_id
    )
    assert torch.isfinite(loss(model, batch))


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the Python family forward reproduce model/fixtures.json."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    assert (
        manifest["family"] == "conv"
        and manifest["labels"] == LINE_KINDS
        and manifest["fields"] == BIO_LABELS
    )
    assert (
        manifest["tags"] == len(LINE_KINDS) + len(BIO_LABELS)
        and manifest["slots"] == NUM_SLOTS
    )
    model = from_config(manifest)
    model.load_tensors(
        decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest)
    )
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
