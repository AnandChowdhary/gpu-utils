"""Export a family model to ``model/{manifest.json, weights.txt, fixtures.json}``.

Fixture logits are computed by the torch model *after reloading the decoded int6
weights* (exactly what runtime/weights.ts produces), so test/parity.test.ts compares the
TypeScript CPU path against the numbers the runtime will actually load.

Canonical fixture format (one for every package)::

    {"cases": [{"input": str | dict, "rows": [[int]], "logits": [[float]], "pooled": [float] | null}]}
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batch import collate
from .models import TaggerBase
from .qat import set_quant
from .quant import decode_weights
from .quant import export as quant_export


def fixture_case(model: TaggerBase, rows: list[list[int]], slots: int, digits: int = 6) -> dict[str, Any]:
    """Run one sequence through ``model`` (quant off) and return ``{"rows", "logits", "pooled"}``."""
    model.eval()
    if not rows:
        return {"rows": rows, "logits": [], "pooled": None if not model.pooled_out else [0.0] * model.pooled_out}
    with torch.no_grad():
        r, m = collate([rows], slots, model.padding_id)
        out = model(r, m)
    tags = out["tags"]
    pooled = out["pooled"]
    assert tags is not None
    return {
        "rows": rows,
        "logits": [[round(float(v), digits) for v in row] for row in tags[0].tolist()],
        "pooled": None if pooled is None else [round(float(v), digits) for v in pooled[0].tolist()],
    }


def export_package(
    model: TaggerBase,
    out_dir: str | Path,
    labels: list[str],
    extra: dict[str, Any],
    fixtures: list[dict[str, Any]],
    *,
    slots: int | None = None,
) -> dict[str, Any]:
    """Write manifest.json + weights.txt (int6) + fixtures.json. Returns the manifest.

    ``fixtures`` items need ``input`` (str or dict) and ``rows`` ([[int]] feature rows); any
    other keys are copied through. ``extra`` is merged into the manifest (``name`` at least).
    """
    out = Path(out_dir)
    if slots is None:
        slots = max((len(r) for case in fixtures for r in case["rows"]), default=1)
    config = model.config()
    manifest = quant_export(
        model.tensors(),
        out,
        {"name": extra.get("name", "model"), **config, "slots": slots, "labels": labels, **{k: v for k, v in extra.items() if k != "name"}},
    )
    # Reload the decoded int6 weights so fixtures reflect exactly what ships.
    ref = copy.deepcopy(model)
    ref.load_tensors(decode_weights((out / "weights.txt").read_text(), manifest))
    set_quant(ref, False)
    cases = []
    for fx in fixtures:
        case = {"input": fx["input"], **fixture_case(ref, fx["rows"], slots)}
        for k, v in fx.items():
            if k not in case:
                case[k] = v
        cases.append(case)
    (out / "fixtures.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False) + "\n")
    return manifest


def check_fixtures(model: TaggerBase, out_dir: str | Path, tolerance: float = 1e-4) -> float:
    """Max |Δ| between the model with QAT on and fixtures.json (should be ~0 after export)."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    cases = json.loads((out / "fixtures.json").read_text())["cases"]
    set_quant(model, True)
    worst = 0.0
    for case in cases:
        got = fixture_case(model, case["rows"], manifest["slots"], digits=9)
        if case["rows"]:
            worst = max(worst, float(np.abs(np.asarray(got["logits"]) - np.asarray(case["logits"])).max()))
        if case["pooled"] is not None:
            worst = max(worst, float(np.abs(np.asarray(got["pooled"]) - np.asarray(case["pooled"])).max()))
    if worst > tolerance:
        raise AssertionError(f"fixtures drift: max |Δ| = {worst:.2e}")
    return worst
