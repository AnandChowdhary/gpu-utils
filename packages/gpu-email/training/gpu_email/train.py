"""Train gpu-email with int6 quantization-aware training through the shared loop.

    uv run python -m gpu_email.train [--epochs 3] [--minutes 18] [--seed 0] [--run default]

Reads data/cache/{train,heldout}.npz written by `python -m gpu_email.data` and writes
runs/<run>/{best.pt,last.pt,history.json}. CPU only, 2 threads by default. The loss is
cross-entropy on the line-kind columns plus class-weighted cross-entropy on the BIO
columns; the best checkpoint is the one with the highest mean of held-out line-kind token
accuracy and BIO span F1 (both measured with fake quantization on).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train
from gpu_utils_training.metrics import bio_to_spans, span_prf, token_accuracy
from gpu_utils_training.models import ConvTagger
from torch import Tensor, nn
from torch.nn import functional as F

from gpu_email.data import CACHE_DIR, Encoded
from gpu_email.features import BIO_LABELS, NUM_SLOTS
from gpu_email.model import N_BIO, N_KINDS, build

RUNS = Path(__file__).resolve().parents[1] / "runs"
BIO_WEIGHTS = torch.ones(N_BIO)
BIO_WEIGHTS[0] = 0.35  # "O" dominates; the old package used the same down-weighting

Batch = tuple[Tensor, Tensor, Tensor, Tensor]


def collate_examples(encoded: list[Encoded], padding_id: int) -> Batch:
    """``(rows [B, T, slots] long, mask [B, T] bool, kinds [B, T], bio [B, T])``; -100 = ignore."""
    n = len(encoded)
    t = max(1, max(len(e.kinds) for e in encoded))
    rows = np.full((n, t, NUM_SLOTS), padding_id, dtype=np.int64)
    mask = np.zeros((n, t), dtype=bool)
    kinds = np.full((n, t), -100, dtype=np.int64)
    bio = np.full((n, t), -100, dtype=np.int64)
    for b, e in enumerate(encoded):
        m = len(e.kinds)
        rows[b, :m] = e.rows
        mask[b, :m] = True
        k = e.kinds.astype(np.int64)
        kinds[b, :m] = np.where(k < 0, -100, k)
        bio[b, :m] = e.bio
    return (
        torch.from_numpy(rows),
        torch.from_numpy(mask),
        torch.from_numpy(kinds),
        torch.from_numpy(bio),
    )


class Split:
    """One cached split (data/cache/*.npz) with length-bucketed, token-budgeted batching."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.rows = z["rows"]
        self.kinds = z["kinds"]
        self.bio = z["bio"]
        self.offsets = z["offsets"]
        self.n = len(self.offsets) - 1
        self.lengths = np.diff(self.offsets)

    def example(self, i: int) -> Encoded:
        s, e = self.offsets[i], self.offsets[i + 1]
        return Encoded(self.rows[s:e], self.kinds[s:e], self.bio[s:e])

    def batch(self, idx: np.ndarray, padding_id: int) -> Batch:
        return collate_examples([self.example(int(i)) for i in idx], padding_id)

    def batches(
        self, rng: np.random.Generator, batch_size: int, max_tokens: int = 12000
    ) -> list[np.ndarray]:
        """Length-bucketed index chunks with a token budget so long emails get smaller batches."""
        order = np.argsort(self.lengths + rng.integers(0, 40, self.n))
        chunks: list[np.ndarray] = []
        i = 0
        while i < self.n:
            length = int(self.lengths[order[i]])
            j = i
            while (
                j < self.n
                and (j - i) < batch_size
                and (j - i + 1) * max(length, int(self.lengths[order[j]])) <= max_tokens
            ):
                length = max(length, int(self.lengths[order[j]]))
                j += 1
            j = max(j, i + 1)
            chunks.append(order[i:j])
            i = j
        rng.shuffle(chunks)
        return chunks


def loss(model: nn.Module, batch: Batch) -> Tensor:
    rows, mask, kinds, bio = batch
    tags = model(rows, mask)["tags"]
    loss_k = F.cross_entropy(
        tags[..., :N_KINDS].reshape(-1, N_KINDS), kinds.reshape(-1), ignore_index=-100
    )
    loss_b = F.cross_entropy(
        tags[..., N_KINDS:].reshape(-1, N_BIO),
        bio.reshape(-1),
        ignore_index=-100,
        weight=BIO_WEIGHTS,
    )
    return loss_k + loss_b


@torch.no_grad()
def evaluate(
    model: ConvTagger, split: Split, max_emails: int = 1500, batch_size: int = 32
) -> dict[str, float]:
    """Token-level line-kind accuracy and BIO span F1 on the first ``max_emails`` of ``split``."""
    model.eval()
    idx_all = np.arange(min(split.n, max_emails))
    kind_p: list[np.ndarray] = []
    kind_g: list[np.ndarray] = []
    kind_m: list[np.ndarray] = []
    pred_spans = []
    gold_spans = []
    for i in range(0, len(idx_all), batch_size):
        rows, mask, kinds, bio = split.batch(
            idx_all[i : i + batch_size], model.padding_id
        )
        tags = model(rows, mask)["tags"]
        kp = tags[..., :N_KINDS].argmax(-1).numpy()
        bp = tags[..., N_KINDS:].argmax(-1).numpy()
        kind_p.append(kp.ravel())
        kind_g.append(kinds.numpy().ravel())
        kind_m.append((kinds.numpy() >= 0).ravel())
        m = mask.numpy()
        g = bio.numpy()
        for b in range(len(rows)):
            n = int(m[b].sum())
            pred_spans.append(bio_to_spans([BIO_LABELS[t] for t in bp[b, :n]]))
            gold_spans.append(bio_to_spans([BIO_LABELS[t] for t in g[b, :n]]))
    acc = token_accuracy(
        np.concatenate(kind_p), np.concatenate(kind_g), np.concatenate(kind_m)
    )
    micro = span_prf(pred_spans, gold_spans)["micro"]
    assert isinstance(micro, dict)
    model.train()
    return {
        "kind_token_acc": acc,
        "bio_span_f1": micro["f1"],
        "score": 0.5 * (acc + micro["f1"]),
    }


def main() -> None:
    args = training_parser(
        "Train gpu-email", epochs=3, lr=2.5e-3, batch=24, minutes=18.0
    ).parse_args()
    train_set = Split(CACHE_DIR / "train.npz")
    heldout = Split(CACHE_DIR / "heldout.npz")
    model = build()
    print(
        f"parameters: {model.parameter_count():,}  train emails: {train_set.n}  tokens: {int(train_set.lengths.sum()):,}  held-out: {heldout.n}"
    )
    result = train(
        model,
        lambda _epoch, rng: [
            train_set.batch(idx, model.padding_id)
            for idx in train_set.batches(rng, args.batch)
        ],
        lambda m: evaluate(m, heldout),  # type: ignore[arg-type]
        loss=loss,
        out_dir=run_dir(__file__, args),
        select="score",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()
