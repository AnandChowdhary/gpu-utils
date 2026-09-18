import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.quant import decode_weights

from gpu_cite.features import WIDTH, featurize
from gpu_cite.labels import NAMEPARTS, ROLES, TAGS, TYPES
from gpu_cite.model import build, crf_nll

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
REF = "Smith, J., & Doe, A. B. (2019). A study of things. Journal of Stuff, 12(3), 45–67."


def test_forward_shapes_and_param_budget() -> None:
    model = build()
    rows, mask = collate([featurize(REF), featurize("x")], WIDTH, model.padding_id)
    out = model(rows, mask)
    assert out["tags"].shape == (2, rows.shape[1], len(TAGS) + len(NAMEPARTS))
    assert out["pooled"].shape == (2, len(TYPES))
    assert 40_000 <= model.parameter_count() <= 100_000
    assert model.tensor_names()[-1] == "trans"
    assert tuple(model.tensors()["trans"].shape) == (len(TAGS), len(TAGS))


def test_padding_does_not_change_logits() -> None:
    torch.manual_seed(1)
    model = build()
    rows = featurize(REF)
    alone = model(*collate([rows], WIDTH, model.padding_id))
    padded = model(*collate([rows, rows + rows], WIDTH, model.padding_id))  # first sequence is padded
    assert torch.allclose(alone["tags"][0], padded["tags"][0, : len(rows)], atol=1e-5)
    assert torch.allclose(alone["pooled"][0], padded["pooled"][0], atol=1e-5)


def test_crf_loss_and_constrained_viterbi() -> None:
    """The shared Viterbi with bio_transitions enforces O -> I-X and B-X -> I-Y bans; the CRF NLL is finite."""
    K, R = len(TAGS), len(ROLES)
    i_title, b_title = 1 + R + 1, 2
    em = np.zeros((4, K))
    em[0, 0] = 5.0
    em[1, i_title] = 5.0  # I-TITLE strongly preferred but illegal after O
    em[2, b_title] = 5.0
    em[3, i_title] = 5.0  # legal after B-TITLE
    start = em.copy()
    start[0] += bio_start_mask(TAGS)
    path = viterbi(start, bio_transitions(TAGS))
    assert path[1] != i_title
    assert path[2:] == [b_title, i_title]
    nll = crf_nll(torch.as_tensor(em, dtype=torch.float32).unsqueeze(0), torch.zeros(K, K), torch.tensor([path]), torch.ones(1, 4, dtype=torch.bool))
    assert torch.isfinite(nll) and nll.item() >= 0.0


def test_exported_model_reproduces_fixtures() -> None:
    """The decoded int6 weights + the family forward reproduce model/fixtures.json (tags and pooled)."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    assert manifest["family"] == "scan" and manifest["paddingId"] == 0 and manifest["slots"] == WIDTH
    assert manifest["tags"] == len(TAGS) + len(NAMEPARTS) and manifest["pooled"] == len(TYPES)
    model = build(hidden=manifest["hidden"], head=manifest["head"])
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
