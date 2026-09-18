"""Alternative design the brief asks us to compare: a mean-pooled multi-label head over the
class vocabulary, one prediction per gold segment (no tagging, no compiler).

    uv run python -m gpu_tailwind.baseline_multilabel [--minutes 4]

Reports exact-set match per example on the held-out set so it can be compared with the
tagger + compiler numbers in training/runs/eval.json. Classes seen fewer than MIN_FREQ
times in training are unreachable for this head by construction (arbitrary values such
as p-[13px] or text-[10px] have effectively unbounded vocabulary).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import torch
from torch import nn

from .features import ROWS, featurize
from .batches import load
from .evaluate import RUNS

MIN_FREQ = 20
D = 48
H = 64


def segments(row: dict) -> list[tuple[list[list[int]], list[str]]]:
    """Split a gold row into segments (by SEP tokens and boundary flags); classes are
    assigned to segments by variant prefix heuristics is impossible without the compiler,
    so we train on whole examples: one bag of classes per phrase."""
    feats = featurize(row["text"])
    return [(feats, row["classes"])]


class Bag(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(ROWS, D, mode="sum")
        self.h = nn.Linear(D, H)
        self.out = nn.Linear(H, k)

    def forward(
        self, ids: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        e = self.emb(ids, offsets) / counts.unsqueeze(1)
        return self.out(torch.relu(self.h(e)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=60000)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = random.Random(0)
    train = load("train", args.limit)
    heldout = load("heldout", 3000)
    freq = Counter(c for r in train for c in r["classes"])
    vocab = sorted(c for c, n in freq.items() if n >= MIN_FREQ)
    index = {c: i for i, c in enumerate(vocab)}
    unreachable = sum(1 for r in heldout for c in r["classes"] if c not in index)
    total = sum(len(r["classes"]) for r in heldout)
    print(
        f"vocab {len(vocab)} classes (freq >= {MIN_FREQ}); held-out classes outside vocab: {unreachable}/{total} = {unreachable / total:.1%}"
    )

    def encode(rows: list[dict]):
        out = []
        for r in rows:
            ids = [i for row in featurize(r["text"]) for i in row]
            y = torch.zeros(len(vocab))
            for c in r["classes"]:
                if c in index:
                    y[index[c]] = 1
            out.append((ids, y, r["classes"]))
        return out

    tr = encode(train)
    he = encode(heldout)
    model = Bag(len(vocab))
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    start = time.time()
    step = 0
    bsz = 128
    while (time.time() - start) / 60 < args.minutes:
        rng.shuffle(tr)
        for i in range(0, len(tr), bsz):
            batch = tr[i : i + bsz]
            ids = torch.tensor([x for b in batch for x in b[0]])
            offsets = torch.tensor([0] + [len(b[0]) for b in batch[:-1]]).cumsum(0)
            counts = torch.tensor([len(b[0]) / 7 for b in batch])
            y = torch.stack([b[1] for b in batch])
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(ids, offsets, counts), y
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(
                    f"step {step} loss {loss.item():.4f} {time.time() - start:.0f}s",
                    flush=True,
                )
            if (time.time() - start) / 60 >= args.minutes:
                break
    model.eval()
    exact = tp = fp = fn = 0
    with torch.no_grad():
        for ids, _, gold in he:
            logits = model(
                torch.tensor(ids), torch.tensor([0]), torch.tensor([len(ids) / 7])
            )[0]
            pred = {vocab[i] for i in (logits > 0).nonzero().flatten().tolist()}
            g = set(gold)
            exact += int(pred == g)
            tp += len(pred & g)
            fp += len(pred - g)
            fn += len(g - pred)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    result = {
        "params": params,
        "vocab": len(vocab),
        "unreachable_class_rate": unreachable / total,
        "exact": exact / len(he),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / max(1e-9, p + r),
        "steps": step,
        "minutes": (time.time() - start) / 60,
    }
    print(json.dumps(result, indent=2))
    RUNS.mkdir(exist_ok=True)
    (RUNS / "baseline_multilabel.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
