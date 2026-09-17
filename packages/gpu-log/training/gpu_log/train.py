"""Train gpu-log with int6 quantization-aware training. `uv run python -m gpu_log.train`.

Default config: 160K synthetic lines, 5 epochs, QAT from epoch 2, ~15 minutes on 2 CPU threads.
Writes runs/latest.pt (gitignored). Then run `python -m gpu_log.export`.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import LABELS, Dataset, cached
from .evaluate import evaluate_dataset
from .model import LogTagger

RUNS = Path(__file__).resolve().parents[1] / "runs"


def batches(ds: Dataset, batch_lines: int, rng: random.Random, shuffle: bool = True):
    """Length-bucketed batches: (feats [B, L, F] long, tags [B, L] long, mask [B, L] float, kinds [B] long)."""
    n = len(ds)
    lengths = np.diff(ds.offsets)
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)  # type: ignore[arg-type]
    chunk = batch_lines * 50
    out = []
    for c in range(0, n, chunk):
        idx = order[c : c + chunk]
        idx = idx[np.argsort(lengths[idx], kind="stable")]
        for b in range(0, len(idx), batch_lines):
            out.append(idx[b : b + batch_lines])
    if shuffle:
        rng.shuffle(out)
    for idx in out:
        L = int(max(1, lengths[idx].max()))
        feats = np.zeros((len(idx), L, ds.features.shape[1]), dtype=np.int64)
        tags = np.zeros((len(idx), L), dtype=np.int64)
        mask = np.zeros((len(idx), L), dtype=np.float32)
        for j, i in enumerate(idx):
            s, e = ds.offsets[i], ds.offsets[i + 1]
            feats[j, : e - s] = ds.features[s:e]
            tags[j, : e - s] = ds.tags[s:e]
            mask[j, : e - s] = 1.0
        yield torch.from_numpy(feats), torch.from_numpy(tags), torch.from_numpy(mask), torch.from_numpy(ds.kinds[idx].astype(np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=160_000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--qat-from", type=int, default=1, help="epoch index from which fake-quant is on")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=RUNS / "latest.pt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = cached("train", args.lines, 1)
    heldout = cached("heldout", 12_000, 2)
    model = LogTagger()
    print(f"params: {model.param_count():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.set_quant(epoch >= args.qat_from)
        model.train()
        loss_sum = 0.0
        n = 0
        for feats, tags, mask, kinds in batches(train, args.batch, rng):
            tag_logits, kind_logits = model(feats, mask)
            tl = F.cross_entropy(tag_logits.reshape(-1, len(LABELS)), tags.reshape(-1), reduction="none")
            tl = (tl * mask.reshape(-1)).sum() / mask.sum()
            kl = F.cross_entropy(kind_logits, kinds)
            loss = tl + 0.5 * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss)
            n += 1
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {loss_sum / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        model.set_quant(True)
        metrics = evaluate_dataset(model, heldout)
        print(f"epoch {epoch} done: train loss {loss_sum / max(n, 1):.4f}; held-out (int6) {metrics['summary']} ({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "seed": args.seed, "epochs": args.epochs, "lines": args.lines}, args.out)
    print(f"saved {args.out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
