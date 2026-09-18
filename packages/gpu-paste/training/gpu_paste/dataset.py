"""Turns generated examples into token-level training arrays and caches them."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from gpu_paste.data import LABELS, LEARNED_KINDS, Example, generate, to_json
from gpu_paste.features import FEATURE_COUNT, featurize_tokens
from gpu_utils_training.features import tokenize

RUNS = Path(__file__).resolve().parents[1] / "runs"
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
KIND_ID = {kind: i for i, kind in enumerate(LEARNED_KINDS)}
MAX_TOKENS = 384

Batch = tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray]
"""``(rows [B,T,F] int64, mask [B,T] bool, labels [B,T] int64 (-100 = pad), kinds [B] int64 (-1 = masked), indices)``."""


@dataclass
class Encoded:
    ids: np.ndarray  # [T, FEATURE_COUNT] int32
    labels: np.ndarray  # [T] int64 (span BIO label ids)
    kind: int  # learned kind id or -1 (masked)


def bio_labels(text: str, tokens, spans: list[tuple[int, int, str]]) -> np.ndarray:
    labels = np.zeros(len(tokens), dtype=np.int64)
    for s, e, kind in spans:
        first = True
        for i, t in enumerate(tokens):
            mid = (t.start + t.end) / 2
            if s <= mid < e and t.end > s and t.start < e:
                labels[i] = LABEL_ID[("B-" if first else "I-") + kind]
                first = False
    return labels


def encode(ex: Example) -> Encoded:
    tokens = tokenize(ex.text)[:MAX_TOKENS]
    ids = np.asarray(featurize_tokens(tokens), dtype=np.int32).reshape(len(tokens), FEATURE_COUNT)
    labels = bio_labels(ex.text, tokens, ex.spans)
    kind = KIND_ID[ex.kind] if ex.kind is not None else -1
    return Encoded(ids, labels, kind)


def _encode_json(d: dict) -> tuple[np.ndarray, np.ndarray, int]:
    ex = Example(d["text"], d["kind"], [(s["span"][0], s["span"][1], s["kind"]) for s in d["spans"]])
    e = encode(ex)
    return e.ids, e.labels, e.kind


def build(n: int, seed: int, workers: int = 3, offline: bool = False) -> tuple[list[Encoded], list[Example]]:
    """Generate + encode n examples, cached under training/runs/ by (n, seed)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    cache = RUNS / f"data-{seed}-{n}.npz"
    raw_path = RUNS / f"data-{seed}-{n}.jsonl"
    if cache.exists() and raw_path.exists():
        z = np.load(cache)
        offsets = z["offsets"]
        ids, labels, kinds = z["ids"], z["labels"], z["kinds"]
        encoded = [Encoded(ids[offsets[i] : offsets[i + 1]], labels[offsets[i] : offsets[i + 1]], int(kinds[i])) for i in range(len(kinds))]
        examples = [Example(d["text"], d["kind"], [(s["span"][0], s["span"][1], s["kind"]) for s in d["spans"]]) for d in map(json.loads, raw_path.read_text().splitlines())]
        return encoded, examples
    print(f"[dataset] generating {n} examples (seed {seed})", file=sys.stderr)
    examples = generate(n, seed, offline=offline)
    dicts = [to_json(ex) for ex in examples]
    print("[dataset] encoding", file=sys.stderr)
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_encode_json, dicts, chunksize=256)
    else:
        rows = [_encode_json(d) for d in dicts]
    encoded = [Encoded(a, b, c) for a, b, c in rows]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(e.labels) for e in encoded])
    np.savez_compressed(
        cache,
        offsets=offsets,
        ids=np.concatenate([e.ids for e in encoded]),
        labels=np.concatenate([e.labels for e in encoded]),
        kinds=np.asarray([e.kind for e in encoded], dtype=np.int64),
    )
    raw_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dicts) + "\n")
    return encoded, examples


class Batches:
    """Length-bucketed batches in the ``[B, T, slots]`` layout the model families expect.

    Padded token slots hold ``padding_id`` (exactly what batch.py/batch.ts pack for the
    kernels); padded labels are ``-100`` and masked kinds ``-1`` for ``cross_entropy``.
    Sized (``__len__``) so ``loop.train`` knows the total step count for its cosine schedule.
    """

    def __init__(self, encoded: list[Encoded], batch_size: int, rng: np.random.Generator | None, padding_id: int, shuffle: bool = True) -> None:
        self.encoded = encoded
        self.padding_id = padding_id
        jitter = rng.integers(0, 3, size=len(encoded)) if (shuffle and rng is not None) else np.zeros(len(encoded), dtype=np.int64)
        order = np.argsort([len(e.labels) + int(j) for e, j in zip(encoded, jitter, strict=True)], kind="stable")
        self.groups = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle and rng is not None:
            rng.shuffle(self.groups)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.groups)

    def pack(self, g: np.ndarray) -> Batch:
        t = max(1, max(len(self.encoded[i].labels) for i in g))
        rows = np.full((len(g), t, FEATURE_COUNT), self.padding_id, dtype=np.int64)
        mask = np.zeros((len(g), t), dtype=bool)
        labels = np.full((len(g), t), -100, dtype=np.int64)
        kinds = np.zeros(len(g), dtype=np.int64)
        for j, i in enumerate(g):
            e = self.encoded[i]
            n = len(e.labels)
            rows[j, :n] = e.ids
            mask[j, :n] = True
            labels[j, :n] = e.labels
            kinds[j] = e.kind
        return torch.from_numpy(rows), torch.from_numpy(mask), torch.from_numpy(labels), torch.from_numpy(kinds), g

    def __iter__(self) -> Iterator[Batch]:
        for g in self.groups:
            yield self.pack(g)


def batches(encoded: list[Encoded], batch_size: int, rng: np.random.Generator | None, padding_id: int, shuffle: bool = True) -> Batches:
    return Batches(encoded, batch_size, rng, padding_id, shuffle)
