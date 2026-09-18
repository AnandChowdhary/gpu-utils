"""Train gpu-view with int6 quantization-aware training.

  uv run python -m gpu_view.train [--samples 120000] [--epochs 10] [--seed 0] [--run default]

CPU only, two threads (the box is shared). The default run takes ~15 minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import data, features
from .generate import ROLES
from .model import ViewTagger

RUNS = Path(__file__).resolve().parent.parent / "runs"


def batches(enc: dict[str, np.ndarray], batch: int, rng: np.random.Generator | None):
    n = enc["rows"].shape[0]
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch):
        idx = order[i : i + batch]
        valid = enc["valid"][idx]
        t = int(valid.any(axis=0).nonzero()[0].max()) + 1 if valid.any() else 1
        yield (
            torch.from_numpy(enc["rows"][idx, :t].astype(np.int64)),
            torch.from_numpy(valid[:, :t]),
            torch.from_numpy(enc["roles"][idx, :t]),
            torch.from_numpy(enc["bounds"][idx, :t]),
        )


@torch.no_grad()
def evaluate(model: ViewTagger, enc: dict[str, np.ndarray]) -> dict[str, float]:
    model.eval()
    tok_correct = tok_total = seq_correct = seq_total = b_correct = 0
    for rows, valid, roles, bounds in batches(enc, 512, None):
        role_logits, bound_logit = model(rows, valid)
        pred = role_logits.argmax(-1)
        pb = (bound_logit > 0).float()
        ok = (pred == roles) | ~valid
        okb = (pb == bounds) | ~valid
        tok_correct += int(((pred == roles) & valid).sum())
        b_correct += int(((pb == bounds) & valid).sum())
        tok_total += int(valid.sum())
        seq_correct += int((ok.all(dim=1) & okb.all(dim=1)).sum())
        seq_total += rows.shape[0]
    model.train()
    return {
        "token_accuracy": tok_correct / max(1, tok_total),
        "boundary_accuracy": b_correct / max(1, tok_total),
        "sequence_exact": seq_correct / max(1, seq_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="default")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    corpora = data.build(args.samples, seed=args.seed)
    model = ViewTagger(features.FEATURE_ROWS, len(ROLES))
    print(f"parameters: {model.parameter_count():,}  feature rows: {features.FEATURE_ROWS}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(corpora["train"]["rows"].shape[0] / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.15)
    out_dir = RUNS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.qat = epoch >= 1  # one float epoch, then quantization-aware
        losses = []
        for rows, valid, roles, bounds in batches(corpora["train"], args.batch, rng):
            role_logits, bound_logit = model(rows, valid)
            v = valid.reshape(-1)
            ce = F.cross_entropy(role_logits.reshape(-1, len(ROLES))[v], roles.reshape(-1)[v])
            bce = F.binary_cross_entropy_with_logits(bound_logit.reshape(-1)[v], bounds.reshape(-1)[v], pos_weight=torch.tensor(2.0))
            loss = ce + 0.5 * bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss))
        model.qat = True
        dev = evaluate(model, corpora["dev"])
        transfer = evaluate(model, corpora["transfer"])
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "dev": dev, "transfer": transfer,
               "elapsed_s": round(time.time() - started)}
        history.append(row)
        print(json.dumps(row))
        torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "parameters": model.parameter_count(),
                                                       "history": history}, indent=2))
    print(f"saved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
