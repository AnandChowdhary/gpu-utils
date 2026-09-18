"""Train gpu-cite with int6 quantization-aware training through the shared loop.

    uv run python -m gpu_cite.train [--epochs 4] [--seed 0] [--minutes 19] [--run default]

Default config (seed 0, 4 epochs over 120K synthetic references, batch 128) runs in under
20 minutes on two CPU threads and writes ``runs/<run>/{best.pt,last.pt,history.json}``.
The loss is the linear-chain CRF NLL over the BIO columns + 0.5 x name-part cross-entropy
on name tokens + 0.5 x document-type cross-entropy on the pooled head.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from gpu_utils_training.batch import collate, pad_labels
from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.features import tokenize
from gpu_utils_training.loop import train
from torch import Tensor, nn
from torch.nn import functional as F

from .data import CACHE, read_jsonl
from .evaluate import evaluate_flat
from .features import WIDTH, featurize_tokens
from .model import build, crf_nll, split_tags

Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def featurize_set(rows: list[dict[str, Any]], cache_name: str) -> list[dict[str, Any]]:
    """Attach feature rows (cached as npz keyed by the number of examples)."""
    cache = CACHE / f"{cache_name}_{len(rows)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        lengths = z["lengths"]
        flat = z["rows"]
        off = 0
        for r, n in zip(rows, lengths, strict=True):
            r["rows"] = flat[off : off + n].tolist()
            off += n
        return rows
    parts: list[np.ndarray] = []
    lengths: list[int] = []
    for r in rows:
        feats = featurize_tokens(r["text"], tokenize(r["text"]))
        assert len(feats) == len(r["tags"]), (r["text"], len(feats), len(r["tags"]))
        r["rows"] = feats
        parts.append(np.asarray(feats, dtype=np.int32).reshape(-1, WIDTH))
        lengths.append(len(feats))
    np.savez_compressed(cache, rows=np.concatenate(parts), lengths=np.asarray(lengths))
    return rows


def collate_examples(batch: list[dict[str, Any]], padding_id: int) -> Batch:
    """(rows [B, T, WIDTH], mask [B, T], BIO tags [B, T], name parts [B, T], types [B])."""
    rows, mask = collate([r["rows"] for r in batch], WIDTH, padding_id)
    max_tokens = rows.shape[1]
    tags = pad_labels([r["tags"] for r in batch], max_tokens, ignore_index=0)
    parts = pad_labels([r["parts"] for r in batch], max_tokens, ignore_index=0)
    types = torch.tensor([r["type"] for r in batch], dtype=torch.long)
    return rows, mask, tags, parts, types


class Batches:
    """Length-bucketed, shuffled batches collated lazily; ``len()`` lets the cosine schedule see the step count."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, rng: np.random.Generator, padding_id: int) -> None:
        self.rows = rows
        self.padding_id = padding_id
        order = np.argsort([len(r["tags"]) + rng.integers(0, 3) for r in rows], kind="stable")
        self.chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        rng.shuffle(self.chunks)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[Batch]:
        for idx in self.chunks:
            yield collate_examples([self.rows[i] for i in idx], self.padding_id)


def loss(model: nn.Module, batch: Batch) -> Tensor:
    rows, mask, tags, parts, types = batch
    out = model(rows, mask)
    emissions, part_logits = split_tags(out["tags"])
    crf = crf_nll(emissions, model.transitions(), tags, mask)  # type: ignore[operator]
    on_name = (tags > 0) & mask
    part_loss = F.cross_entropy(part_logits[on_name], parts[on_name]) if bool(on_name.any()) else emissions.new_zeros(())
    type_loss = F.cross_entropy(out["pooled"], types)
    return crf + 0.5 * part_loss + 0.5 * type_loss


def main() -> None:
    ap = training_parser("Train gpu-cite", epochs=4, lr=3e-3, batch=128, minutes=19)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training examples (smoke tests)")
    args = ap.parse_args()
    train_rows = read_jsonl(CACHE / "train.jsonl.gz")
    held_rows = read_jsonl(CACHE / "heldout.jsonl.gz")
    train_rows = featurize_set(train_rows, "feats_train")
    held_rows = featurize_set(held_rows, "feats_heldout")
    if args.limit:
        train_rows = train_rows[: args.limit]
    model = build(hidden=args.hidden)
    print(f"parameters: {model.parameter_count():,}  train: {len(train_rows)}  held-out: {len(held_rows)}", flush=True)
    out = run_dir(__file__, args)
    result = train(
        model,
        lambda _epoch, rng: Batches(train_rows, args.batch, rng, model.padding_id),
        lambda m: evaluate_flat(m, held_rows),
        loss=loss,
        out_dir=out,
        select="exact_match",
        **loop_kwargs(args),
    )
    summary = {**(result["best"] or {}), "params": model.parameter_count(), "train_examples": len(train_rows), "minutes": result["minutes"]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
