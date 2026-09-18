"""Role-level evaluation: token accuracy, boundary accuracy and sequence exact match
(all roles and boundaries right). Spec-level exact match, including the hand-written
unfamiliar sets, is measured in TypeScript (`pnpm test`, test/eval.test.ts) because the
compiler only exists there.

    uv run python -m gpu_view.evaluate [--run default]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.loop import load_checkpoint
from torch import nn

from . import data
from .generate import ROLES
from .model import build

RUNS = Path(__file__).resolve().parents[1] / "runs"


@torch.no_grad()
def evaluate(model: nn.Module, enc: dict[str, np.ndarray]) -> dict[str, float]:
    k = len(ROLES)
    tok_correct = tok_total = seq_correct = seq_total = b_correct = 0
    for rows, mask, roles, bounds in data.batches(enc, 512, None):
        logits = model(rows, mask)["tags"]
        pred = logits[..., :k].argmax(-1)
        pb = (logits[..., k] > 0).float()
        ok = (pred == roles) | ~mask
        okb = (pb == bounds) | ~mask
        tok_correct += int(((pred == roles) & mask).sum())
        b_correct += int(((pb == bounds) & mask).sum())
        tok_total += int(mask.sum())
        seq_correct += int((ok.all(dim=1) & okb.all(dim=1)).sum())
        seq_total += rows.shape[0]
    return {
        "token_accuracy": tok_correct / max(1, tok_total),
        "boundary_accuracy": b_correct / max(1, tok_total),
        "sequence_exact": seq_correct / max(1, seq_total),
    }


def evaluate_both(model: nn.Module, corpora: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    dev = evaluate(model, corpora["dev"])
    transfer = evaluate(model, corpora["transfer"])
    return {**{f"dev_{k}": v for k, v in dev.items()}, **{f"transfer_{k}": v for k, v in transfer.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--samples", type=int, default=160000)
    args = ap.parse_args()
    model = build()
    load_checkpoint(model, RUNS / args.run / "best.pt")
    model.quant = True
    model.eval()
    corpora = data.build(args.samples, seed=0)
    print(json.dumps(evaluate_both(model, corpora), indent=1))


if __name__ == "__main__":
    main()
