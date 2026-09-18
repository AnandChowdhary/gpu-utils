"""Evaluation: token accuracy, exact match and span P/R/F1 with the same Viterbi/BIO
constraints the TypeScript decoder applies. Report held-out AND an "unfamiliar"
hand-written set in MODEL_CARD.md (add data/unfamiliar.jsonl and load it here).

    uv run python -m __SNAKE__.evaluate [--run default]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.loop import load_checkpoint
from gpu_utils_training.metrics import bio_to_spans, exact_match, format_prf_table, span_prf, token_accuracy
from torch import nn

from .data import Example, encode, generate
from .features import LABELS, SLOTS
from .model import build

RUNS = Path(__file__).resolve().parents[1] / "runs"
TRANSITIONS = bio_transitions(LABELS)
START = bio_start_mask(LABELS)


@torch.no_grad()
def predict(model: nn.Module, rows: list[list[list[int]]], padding_id: int, batch_size: int = 256) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        r, mask = collate(chunk, SLOTS, padding_id)
        logits = model(r, mask)["tags"].numpy()
        for j, seq in enumerate(chunk):
            em = logits[j, : len(seq)].astype(np.float64)
            if len(seq):
                em[0] += START
            out.append(viterbi(em, TRANSITIONS))
    return out


def evaluate(model: nn.Module, examples: list[Example]) -> dict[str, float]:
    rows, gold = encode(examples)
    pred = predict(model, rows, model.padding_id)  # type: ignore[attr-defined]
    spans = span_prf(
        [bio_to_spans([LABELS[i] for i in p]) for p in pred],
        [bio_to_spans([LABELS[i] for i in g]) for g in gold],
    )
    flat_p = np.concatenate([np.asarray(p) for p in pred]) if pred else np.zeros(0)
    flat_g = np.concatenate([np.asarray(g) for g in gold]) if gold else np.zeros(0)
    micro = spans["micro"]
    assert isinstance(micro, dict)
    return {
        "token_accuracy": token_accuracy(flat_p, flat_g),
        "exact_match": exact_match(pred, gold),
        "span_f1": micro["f1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    args = ap.parse_args()
    model = build()
    load_checkpoint(model, RUNS / args.run / "best.pt")
    model.quant = True
    held = generate(2_000, seed=1)
    print(json.dumps(evaluate(model, held), indent=1))
    rows, gold = encode(held)
    pred = predict(model, rows, model.padding_id)
    print(format_prf_table(span_prf([bio_to_spans([LABELS[i] for i in p]) for p in pred], [bio_to_spans([LABELS[i] for i in g]) for g in gold])))


if __name__ == "__main__":
    main()
