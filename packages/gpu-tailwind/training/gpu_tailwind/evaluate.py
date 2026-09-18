"""Tag-level evaluation with the same Viterbi/BIO constraints the TypeScript decoder applies.

    uv run python -m gpu_tailwind.evaluate [--run default]

Writes training/runs/<run>/tag_eval.json. Class-level metrics (exact-set match, per-class
precision/recall on the held-out and unfamiliar sets) need the TypeScript compiler:
`pnpm --filter gpu-tailwind eval`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.loop import load_checkpoint
from gpu_utils_training.metrics import bio_to_spans, exact_match, span_prf, token_accuracy
from torch import nn

from .batches import Encoded, encode, iter_batches, load
from .labels import BOUNDARY, LABELS
from .model import build

RUNS = Path(__file__).resolve().parents[1] / "runs"
TRANSITIONS = bio_transitions(LABELS)
START = bio_start_mask(LABELS)


@torch.no_grad()
def predict(model: nn.Module, data: list[Encoded], padding_id: int, batch_size: int = 256) -> tuple[list[list[int]], list[list[int]]]:
    """Viterbi role ids and boundary flags per sequence."""
    roles: list[list[int]] = []
    bounds: list[list[int]] = []
    for chunk, (rows, mask, _labels, _boundary) in iter_batches(data, batch_size, padding_id):
        tags = model(rows, mask)["tags"].numpy()
        for j, item in enumerate(chunk):
            n = len(item[1])
            em = tags[j, :n, :BOUNDARY].astype(np.float64)
            if n:
                em[0] += START
            roles.append(viterbi(em, TRANSITIONS))
            bounds.append([int(v > 0) for v in tags[j, :n, BOUNDARY]])
    return roles, bounds


def evaluate(model: nn.Module, data: list[Encoded]) -> dict[str, float]:
    pred, bounds = predict(model, data, model.padding_id)  # type: ignore[attr-defined]
    gold = [d[1] for d in data]
    gold_b = [d[2] for d in data]
    spans = span_prf(
        [bio_to_spans([LABELS[i] for i in p]) for p in pred],
        [bio_to_spans([LABELS[i] for i in g]) for g in gold],
    )
    flat_p = np.concatenate([np.asarray(p) for p in pred])
    flat_g = np.concatenate([np.asarray(g) for g in gold])
    flat_pb = np.concatenate([np.asarray(b) for b in bounds])
    flat_gb = np.concatenate([np.asarray(b) for b in gold_b])
    per = spans["per_label"]
    assert isinstance(per, dict)
    out = {
        "token_acc": token_accuracy(flat_p, flat_g),
        "seq_acc": exact_match(pred, gold),
        "boundary_acc": float((flat_pb == flat_gb).mean()),
    }
    for label in ("PROP", "VAL", "VAR"):
        for k in ("precision", "recall", "f1"):
            out[f"{label}_{k}"] = float(per.get(label, {}).get(k, 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = build()
    load_checkpoint(model, RUNS / args.run / "best.pt")
    model.quant = True
    model.eval()
    data = encode(load("heldout"))
    result = {"n": len(data), **evaluate(model, data)}
    print(json.dumps(result, indent=1))
    (RUNS / args.run / "tag_eval.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
