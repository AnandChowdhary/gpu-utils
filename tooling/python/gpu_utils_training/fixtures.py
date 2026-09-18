"""Generates the cross-language fixtures under packages/runtime/test/fixtures:

* ``scan_tagger.json`` / ``conv_tagger.json``: a small random-weight model of each family
  (manifest + int6 weights inline) with cases in the canonical fixture format, computed by
  the Python reference on the decoded weights. models.ts must reproduce them at 1e-4 and
  the canonical WGSL kernels must too (tooling/python/tests/test_wgsl_parity.py).
* ``viterbi.json``: emissions/transitions/expected paths for decode.ts vs decode.py.

    uv run python -m gpu_utils_training.fixtures        # rewrite the files
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .decode import bio_transitions, viterbi
from .export import fixture_case
from .models import ConvTagger, ScanTagger, TaggerBase
from .qat import set_quant
from .quant import decode_weights
from .quant import export as quant_export

FIXTURES = Path(__file__).resolve().parents[3] / "packages" / "runtime" / "test" / "fixtures"
SLOTS = 3
FEATURE_ROWS = 64
VITERBI_LABELS = ["O", "B-A", "I-A", "B-B", "I-B"]


def random_rows(rng: np.random.Generator, n_cases: int, max_tokens: int) -> list[list[list[int]]]:
    cases = []
    for i in range(n_cases):
        n = 1 if i == 0 else int(rng.integers(1, max_tokens + 1))
        seq = []
        for _ in range(n):
            k = int(rng.integers(1, SLOTS + 1))
            seq.append([int(v) for v in rng.integers(0, FEATURE_ROWS, size=k)])
        cases.append(seq)
    return cases


def build_models(seed: int = 7) -> dict[str, TaggerBase]:
    torch.manual_seed(seed)
    return {
        "scan": ScanTagger(FEATURE_ROWS, 8, 5, pooled_out=3, scan_layers=2),
        "conv": ConvTagger(FEATURE_ROWS, 6, 8, 3, [1, 2, 4], 4, pooled_out=2),
    }


def family_fixture(model: TaggerBase, n_cases: int = 12, seed: int = 11) -> dict[str, Any]:
    """Export to a temp dir, reload the decoded weights and compute the cases."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        manifest = quant_export(model.tensors(), out, {"name": f"fixture-{model.family}", **model.config(), "slots": SLOTS, "labels": [f"L{i}" for i in range(model.n_tags)]})
        encoded = (out / "weights.txt").read_text()
    ref = copy.deepcopy(model)
    ref.load_tensors(decode_weights(encoded, manifest))
    set_quant(ref, False)
    cases = [{"input": f"case {i}", **fixture_case(ref, rows, SLOTS)} for i, rows in enumerate(random_rows(rng, n_cases, 12))]
    return {"manifest": manifest, "weights": encoded, "cases": cases}


def viterbi_fixture(seed: int = 3, n_cases: int = 10) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    k = len(VITERBI_LABELS)
    base = bio_transitions(VITERBI_LABELS)
    cases = []
    for i in range(n_cases):
        n = int(rng.integers(1, 9))
        em = np.round(rng.normal(0, 2, size=(n, k)), 3)
        tr = base + np.round(rng.normal(0, 0.5, size=(k, k)), 3).astype(np.float32) * (base == 0)
        if i % 3 == 0:  # exact ties: both sides must pick the lowest index
            em[:, 1] = em[:, 0]
        cases.append({"emissions": em.tolist(), "transitions": tr.tolist(), "path": viterbi(em, tr)})
    return {"labels": VITERBI_LABELS, "cases": cases}


def generate() -> dict[str, dict[str, Any]]:
    models = build_models()
    return {
        "scan_tagger.json": family_fixture(models["scan"]),
        "conv_tagger.json": family_fixture(models["conv"]),
        "viterbi.json": viterbi_fixture(),
    }


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, data in generate().items():
        (FIXTURES / name).write_text(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"wrote {FIXTURES / name}")


if __name__ == "__main__":
    main()
