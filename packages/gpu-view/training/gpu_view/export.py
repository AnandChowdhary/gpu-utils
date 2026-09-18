"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_view.export [--run default] [--random]

fixtures.json uses the canonical format ({"cases": [{input, rows, logits, pooled}]}) with
`input = {"text", "schema"}`, computed from the decoded int6 weights, so test/parity.test.ts
and training/tests/test_wgsl.py compare against exactly what the runtime loads.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch
from gpu_utils_training.export import export_package
from gpu_utils_training.loop import load_checkpoint

from . import features
from .generate import ROLES, dataset
from .model import build

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODEL_DIR = HERE.parent.parent / "model"

HAND_WRITTEN = [
    ("total revenue by region this quarter, top 10, as a bar chart", {"fields": [
        {"name": "revenue", "kind": "number"}, {"name": "region", "kind": "enum", "values": ["EMEA", "APAC"]},
        {"name": "closed_at", "kind": "date"}]}),
    ("open issues assigned to me sorted by priority", {"fields": [
        {"name": "status", "kind": "enum", "values": ["open", "closed"]},
        {"name": "assignee", "kind": "text", "aliases": ["assigned"]},
        {"name": "priority", "kind": "enum", "values": ["low", "high"]}]}),
    ("cheapest first, taller than 50 cm", {"fields": [
        {"name": "price", "kind": "number", "aliases": ["cheap"]}, {"name": "height", "kind": "number", "aliases": ["tall"]}]}),
    ("", {"fields": [{"name": "x", "kind": "text"}]}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--random", action="store_true", help="export an untrained model")
    args = ap.parse_args()
    torch.manual_seed(0)
    model = build()
    checkpoint: dict[str, object] = {"run": None, "date": date.today().isoformat()}
    if not args.random:
        ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
        checkpoint = {"run": args.run, "epoch": ckpt.get("epoch"), "seed": ckpt.get("seed"), "metrics": ckpt.get("metrics"), "date": date.today().isoformat()}
    inputs = [(e.text, e.schema) for e in dataset("train", 14, seed=31) + dataset("eval", 10, seed=32)] + HAND_WRITTEN
    fixtures = []
    for text, schema in inputs:
        toks, rows = features.featurize(text, schema)
        fixtures.append({"input": {"text": text, "schema": schema}, "rows": rows, "tokens": [t.text for t in toks]})
    extra = {
        "name": "gpu-view",
        "boundary": True,  # tags = roles + 1: the last column is the clause-boundary logit
        "blocks": [{"name": n, "offset": features.OFFSET[n], "size": s} for n, s in features.BLOCKS],
        "checkpoint": checkpoint,
    }
    manifest = export_package(model, MODEL_DIR, ROLES, extra, fixtures, slots=features.SLOTS)
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("tensors", "blocks")}, indent=1))


if __name__ == "__main__":
    main()
