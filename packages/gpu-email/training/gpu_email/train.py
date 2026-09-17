"""Train gpu-email with quantization-aware training.

    uv run python -m gpu_email.train [--minutes 16] [--epochs 3] [--seed 0]

Reads data/cache/{train,heldout}.npz written by `python -m gpu_email.data`, writes
runs/latest.pt (gitignored) and prints token-level metrics on the held-out set.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gpu_email.data import CACHE_DIR
from gpu_email.features import BIO_LABELS, LINE_KINDS, NUM_SLOTS
from gpu_email.model import EmailTagger

RUNS = Path(__file__).resolve().parents[1] / "runs"


class Split:
    def __init__(self, path: Path):
        z = np.load(path)
        self.rows = z["rows"]
        self.kinds = z["kinds"].astype(np.int64)
        self.bio = z["bio"].astype(np.int64)
        self.offsets = z["offsets"]
        self.n = len(self.offsets) - 1
        self.lengths = np.diff(self.offsets)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = int(self.lengths[idx].max())
        B = len(idx)
        ids = np.zeros((B, L, NUM_SLOTS), dtype=np.int64)
        mask = np.zeros((B, L), dtype=np.float32)
        kinds = np.full((B, L), -100, dtype=np.int64)
        bio = np.full((B, L), -100, dtype=np.int64)
        for b, i in enumerate(idx):
            s, e = self.offsets[i], self.offsets[i + 1]
            n = e - s
            ids[b, :n] = self.rows[s:e]
            mask[b, :n] = 1
            k = self.kinds[s:e]
            kinds[b, :n] = np.where(k < 0, -100, k)
            bio[b, :n] = self.bio[s:e]
        return torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(kinds), torch.from_numpy(bio)

    def batches(self, rng: np.random.Generator, batch_size: int, max_tokens: int = 12000):
        """Length-bucketed batches with a token budget so long emails get smaller batches."""
        order = np.argsort(self.lengths + rng.integers(0, 40, self.n))
        chunks: list[np.ndarray] = []
        i = 0
        while i < self.n:
            L = int(self.lengths[order[i]])
            j = i
            while j < self.n and (j - i) < batch_size and (j - i + 1) * max(L, int(self.lengths[order[j]])) <= max_tokens:
                L = max(L, int(self.lengths[order[j]]))
                j += 1
            j = max(j, i + 1)
            chunks.append(order[i:j])
            i = j
        rng.shuffle(chunks)
        return chunks


def evaluate(model: EmailTagger, split: Split, max_emails: int = 1500) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(0)
    idx_all = np.arange(min(split.n, max_emails))
    kind_correct = kind_total = 0
    bio_tp = np.zeros(len(BIO_LABELS))
    bio_fp = np.zeros(len(BIO_LABELS))
    bio_fn = np.zeros(len(BIO_LABELS))
    with torch.no_grad():
        for i in range(0, len(idx_all), 32):
            ids, mask, kinds, bio = split.batch(idx_all[i : i + 32])
            k_logits, b_logits = model(ids, mask)
            kp = k_logits.argmax(-1)
            valid = kinds >= 0
            kind_correct += int(((kp == kinds) & valid).sum())
            kind_total += int(valid.sum())
            bp = b_logits.argmax(-1)
            v = bio >= 0
            for c in range(1, len(BIO_LABELS)):
                bio_tp[c] += int(((bp == c) & (bio == c) & v).sum())
                bio_fp[c] += int(((bp == c) & (bio != c) & v).sum())
                bio_fn[c] += int(((bp != c) & (bio == c) & v).sum())
    del rng
    tp, fp, fn = bio_tp[1:].sum(), bio_fp[1:].sum(), bio_fn[1:].sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    model.train()
    return {"kind_token_acc": kind_correct / max(1, kind_total), "bio_token_f1": float(f1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--qat-from", type=float, default=0.4, help="fraction of steps after which QAT is on")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = Split(CACHE_DIR / "train.npz")
    heldout = Split(CACHE_DIR / "heldout.npz")
    model = EmailTagger(args.dim, args.hidden)
    print(f"params: {model.num_params()}  train emails: {train.n}  tokens: {int(train.lengths.sum())}", file=sys.stderr)

    steps_per_epoch = len(train.batches(np.random.default_rng(1), args.batch))
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=50)
    bio_weights = torch.ones(len(BIO_LABELS))
    bio_weights[0] = 0.35
    ce_kind = nn.CrossEntropyLoss(ignore_index=-100)
    ce_bio = nn.CrossEntropyLoss(ignore_index=-100, weight=bio_weights)

    t0 = time.time()
    step = 0
    deadline = t0 + args.minutes * 60
    done = False
    for epoch in range(args.epochs):
        for idx in train.batches(rng, args.batch):
            if step >= int(args.qat_from * total_steps) and not model.qat:
                model.qat = True
                print(f"step {step}: QAT on", file=sys.stderr)
            ids, mask, kinds, bio = train.batch(idx)
            k_logits, b_logits = model(ids, mask)
            loss_k = ce_kind(k_logits.reshape(-1, len(LINE_KINDS)), kinds.reshape(-1))
            loss_b = ce_bio(b_logits.reshape(-1, len(BIO_LABELS)), bio.reshape(-1))
            loss = loss_k + loss_b
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_steps - 1:
                sched.step()
            step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"ep {epoch} step {step}/{total_steps} loss {loss.item():.3f} (kind {loss_k.item():.3f} bio {loss_b.item():.3f}) lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f}min", file=sys.stderr)
            if time.time() > deadline:
                print("time budget reached", file=sys.stderr)
                done = True
                break
        m = evaluate(model, heldout)
        print(f"epoch {epoch}: heldout {m}", file=sys.stderr)
        if done:
            break
    if not model.qat:
        model.qat = True
    m = evaluate(model, heldout)
    print(f"final (QAT): heldout {m}  elapsed {(time.time() - t0) / 60:.1f}min", file=sys.stderr)
    RUNS.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": args.dim, "hidden": args.hidden, "dilations": model.dilations, "metrics": m, "seed": args.seed, "steps": step, "epochs": args.epochs}, RUNS / "latest.pt")
    print(f"saved {RUNS / 'latest.pt'}", file=sys.stderr)
    assert not math.isnan(loss.item())


if __name__ == "__main__":
    main()
