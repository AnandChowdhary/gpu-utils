"""Synthetic data. Replace `generate` with the real task's grammar/teacher; keep the
`Example` shape (text + one BIO tag per token) so train/evaluate/export stay unchanged.

The placeholder task tags Title-case runs as NAME spans, which a fresh model learns in
one short epoch; it exists so the scaffold trains, exports and passes parity end to end.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from gpu_utils_training.batch import collate, pad_labels
from gpu_utils_training.features import SHAPE_TITLE, tokenize
from torch import Tensor, nn
from torch.nn import functional as F

from .features import LABELS, SLOTS, featurize_tokens

WORDS = "call email meet with and the a to at on about please schedule remind me later today tomorrow".split()
NAMES = "Ada Grace Linus Alan Margaret Tim Ken Dennis Barbara Radia Anita Frances".split()
LABEL_ID = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class Example:
    text: str
    tags: list[str]


def _tag(text: str) -> list[str]:
    tags: list[str] = []
    in_name = False
    tokens = tokenize(text)
    for i, t in enumerate(tokens):
        if t.shape == SHAPE_TITLE:
            tags.append("I-NAME" if in_name else "B-NAME")
            in_name = True
        elif t.text == " " and in_name and i + 1 < len(tokens) and tokens[i + 1].shape == SHAPE_TITLE:
            tags.append("I-NAME")
        else:
            tags.append("O")
            in_name = False
    return tags


def generate(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for _ in range(n):
        words = [rng.choice(WORDS) for _ in range(rng.randint(2, 8))]
        for _ in range(rng.randint(0, 2)):
            k = rng.randint(0, len(words))
            words[k:k] = [rng.choice(NAMES) for _ in range(rng.randint(1, 2))]
        text = " ".join(words)
        if rng.random() < 0.3:
            text += rng.choice([".", "!", ","])
        out.append(Example(text, _tag(text)))
    return out


def encode(examples: list[Example]) -> tuple[list[list[list[int]]], list[list[int]]]:
    rows = [featurize_tokens(tokenize(e.text)) for e in examples]
    labels = [[LABEL_ID[t] for t in e.tags] for e in examples]
    return rows, labels


def batches(examples: list[Example], batch_size: int, rng: np.random.Generator, padding_id: int) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Length-bucketed shuffled batches of (rows, mask, label ids)."""
    rows, labels = encode(examples)
    order = np.argsort([len(r) + rng.integers(0, 3) for r in rows], kind="stable")
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)  # type: ignore[arg-type]
    for idx in chunks:
        r, mask = collate([rows[i] for i in idx], SLOTS, padding_id)
        yield r, mask, pad_labels([labels[i] for i in idx], r.shape[1])


def loss(model: nn.Module, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    rows, mask, labels = batch
    logits = model(rows, mask)["tags"]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def main() -> None:
    for e in generate(5, seed=0):
        print(e.text, "→", " ".join(e.tags))
    print("(synthetic only: nothing to download)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
