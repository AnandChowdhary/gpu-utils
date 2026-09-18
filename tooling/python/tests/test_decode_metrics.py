import json
from pathlib import Path

import numpy as np

from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.metrics import bio_to_spans, confusion, exact_match, format_prf_table, span_prf, token_accuracy

FIXTURES = Path(__file__).resolve().parents[3] / "packages/runtime/test/fixtures"


def test_viterbi_matches_fixture() -> None:
    data = json.loads((FIXTURES / "viterbi.json").read_text())
    for case in data["cases"]:
        assert viterbi(np.array(case["emissions"]), np.array(case["transitions"])) == case["path"]


def test_viterbi_forbids_o_to_i() -> None:
    labels = ["O", "B-X", "I-X"]
    em = np.array([[5, 0, 0], [0, 1, 4], [0, 0, 4]], dtype=np.float64)
    assert viterbi(em, bio_transitions(labels)) == [0, 1, 2]
    assert viterbi(np.zeros((0, 3)), bio_transitions(labels)) == []


def test_bio_transitions_table() -> None:
    labels = ["O", "B-A", "I-A", "B-B", "I-B"]
    t = bio_transitions(labels)
    assert t[0, 2] < 0 and t[1, 2] == 0 and t[2, 2] == 0 and t[3, 2] < 0 and t[4, 2] < 0
    assert t[:, 0].tolist() == [0] * 5 and t[:, 1].tolist() == [0] * 5
    assert bio_start_mask(labels).tolist() == [0, 0, -1e9, 0, -1e9]


def test_bio_to_spans_and_prf() -> None:
    assert bio_to_spans(["B-A", "I-A", "O", "I-B", "I-B", "B-B"]) == [("A", 0, 2), ("B", 3, 5), ("B", 5, 6)]
    gold = [[("A", 0, 2), ("B", 3, 5)]]
    pred = [[("A", 0, 2), ("B", 3, 4)]]
    r = span_prf(pred, gold)
    assert r["micro"]["precision"] == 0.5 and r["micro"]["recall"] == 0.5
    assert r["per_label"]["A"]["f1"] == 1.0 and r["per_label"]["B"]["f1"] == 0.0
    assert r["macro"]["f1"] == 0.5
    assert "| A |" in format_prf_table(r)


def test_exact_token_confusion() -> None:
    assert exact_match([[1, 2], [3]], [[1, 2], [4]]) == 0.5
    assert token_accuracy(np.array([1, 2, 3]), np.array([1, 0, 3]), np.array([True, True, False])) == 0.5
    c = confusion(np.array([1, 1, 0]), np.array([1, 0, 0]), 2)
    assert c.tolist() == [[1, 1], [0, 1]]
