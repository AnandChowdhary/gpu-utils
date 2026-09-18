import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.export import check_fixtures
from gpu_utils_training.qat import set_quant

from gpu_log.data import KINDS, LABELS, batches, cached, loss
from gpu_log.export import MODEL_DIR
from gpu_log.features import FEATURE_COUNT, featurize
from gpu_log.model import build


def test_family_shapes_and_budget() -> None:
    model = build()
    assert 100_000 <= model.parameter_count() <= 250_000
    rows, mask = collate([featurize("2024-01-15 INFO hello world k=v")[1], featurize("\tat a.b.C.d(C.java:12)")[1]], FEATURE_COUNT, model.padding_id)
    out = model(rows, mask)
    assert out["tags"].shape == (2, rows.shape[1], len(LABELS))
    assert out["pooled"].shape == (2, len(KINDS))


def test_batches_and_loss() -> None:
    ds = cached("smoke", 300, 5)
    model = build()
    bs = batches(ds, 16, np.random.default_rng(0), model.padding_id)
    assert sum(len(b[3]) for b in bs) == len(ds)
    value = loss(model, bs[0])
    assert torch.isfinite(value)


def test_exported_fixtures_match_model() -> None:
    """The shipped model/ reproduces fixtures.json (QAT on = decoded int6 weights)."""
    from gpu_utils_training.models import from_config
    from gpu_utils_training.quant import decode_weights
    import json

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    model = from_config(manifest)
    model.load_tensors(decode_weights((MODEL_DIR / "weights.txt").read_text(), manifest))
    set_quant(model, False)
    assert check_fixtures(model, MODEL_DIR) < 1e-4
