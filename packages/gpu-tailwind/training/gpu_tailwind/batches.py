"""Encoding, batching and the loss for gpu_utils_training.loop.train.

A batch is ``(rows, mask, labels, boundary)``; the tag head has len(LABELS) role columns
trained with cross-entropy plus one boundary column trained with BCE.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate, pad_labels
from torch import Tensor, nn
from torch.nn import functional as F

from .features import WIDTH, featurize
from .labels import BOUNDARY, LABEL_ID

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
BOUNDARY_WEIGHT = 0.5

Encoded = tuple[list[list[int]], list[int], list[int]]  # rows, label ids, boundary flags


def load(name: str, limit: int | None = None) -> list[dict]:
    path = CACHE / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `uv run python -m gpu_tailwind.data` first")
    rows = [json.loads(line) for line in path.open()]
    return rows[:limit] if limit else rows


def encode(rows: list[dict]) -> list[Encoded]:
    out: list[Encoded] = []
    for r in rows:
        feats = featurize(r["text"])
        assert len(feats) == len(r["labels"]), r["text"]
        out.append((feats, [LABEL_ID[label] for label in r["labels"]], r["boundary"]))
    return out


def make_batch(items: list[Encoded], padding_id: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rows, mask = collate([it[0] for it in items], WIDTH, padding_id)
    labels = pad_labels([it[1] for it in items], rows.shape[1])
    boundary = torch.zeros(rows.shape[0], rows.shape[1])
    for i, it in enumerate(items):
        boundary[i, : len(it[2])] = torch.as_tensor(it[2], dtype=torch.float)
    return rows, mask, labels, boundary


def batches(data: list[Encoded], batch_size: int, rng: np.random.Generator, padding_id: int) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Length-bucketed shuffled batches (a list so the loop can size its cosine schedule)."""
    order = np.argsort([len(d[1]) + rng.integers(0, 3) for d in data], kind="stable")
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)  # type: ignore[arg-type]
    return [make_batch([data[i] for i in idx], padding_id) for idx in chunks]


def iter_batches(data: list[Encoded], batch_size: int, padding_id: int) -> Iterator[tuple[list[Encoded], tuple[Tensor, Tensor, Tensor, Tensor]]]:
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        yield chunk, make_batch(chunk, padding_id)


def loss(model: nn.Module, batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    rows, mask, labels, boundary = batch
    tags = model(rows, mask)["tags"]
    roles = tags[..., :BOUNDARY]
    ce = F.cross_entropy(roles.reshape(-1, roles.shape[-1]), labels.reshape(-1), ignore_index=-100)
    bce = F.binary_cross_entropy_with_logits(tags[..., BOUNDARY], boundary, reduction="none")
    m = mask.to(bce.dtype)
    return ce + BOUNDARY_WEIGHT * (bce * m).sum() / m.sum().clamp(min=1.0)
