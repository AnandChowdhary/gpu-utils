"""Batch packing shared by training, export and the WGSL harness.

Mirrors packages/runtime/src/batch.ts: a batch of variable-length feature-row sequences
becomes ``rows [B, max_tokens, slots]`` (unused slots and padded tokens hold ``padding_id``)
plus ``lengths [B]``. The same layout feeds the torch models (as int64 + bool mask) and the
canonical WGSL kernels (as u32).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def pack_rows(batch: list[list[list[int]]], slots: int, padding_id: int, max_tokens: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pack feature rows into ``(rows [B, T, slots] uint32, lengths [B] uint32)``."""
    n = len(batch)
    t = max_tokens if max_tokens is not None else max(1, max((len(r) for r in batch), default=1))
    rows = np.full((n, t, slots), padding_id, dtype=np.uint32)
    lengths = np.zeros(n, dtype=np.uint32)
    for s, seq in enumerate(batch):
        lengths[s] = len(seq)
        for i, row in enumerate(seq):
            k = min(len(row), slots)
            rows[s, i, :k] = row[:k]
    return rows, lengths


def to_torch(rows: np.ndarray, lengths: np.ndarray) -> tuple[Tensor, Tensor]:
    """``(rows int64 [B, T, S], mask bool [B, T])`` for the model families."""
    r = torch.from_numpy(rows.astype(np.int64))
    mask = torch.arange(rows.shape[1]).unsqueeze(0) < torch.from_numpy(lengths.astype(np.int64)).unsqueeze(1)
    return r, mask


def collate(batch: list[list[list[int]]], slots: int, padding_id: int) -> tuple[Tensor, Tensor]:
    """One-call convenience: feature rows -> ``(rows, mask)`` tensors."""
    return to_torch(*pack_rows(batch, slots, padding_id))


def pad_labels(labels: list[list[int]], max_tokens: int, ignore_index: int = -100) -> Tensor:
    """Per-token label ids padded with ``ignore_index`` for ``cross_entropy``."""
    out = torch.full((len(labels), max_tokens), ignore_index, dtype=torch.long)
    for i, seq in enumerate(labels):
        out[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
    return out
