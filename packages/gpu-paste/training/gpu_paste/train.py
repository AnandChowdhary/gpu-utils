"""Train gpu-paste with int6 quantization-aware training.

    uv run python -m gpu_paste.train [--n 120000] [--epochs 6] [--seed 1]

Writes runs/best.pt (state dict + metrics) and runs/metrics.json. Default run: ~15 min on
two CPU threads.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS
from gpu_paste.dataset import RUNS, batches, build
from gpu_paste.evaluate import evaluate_model
from gpu_paste.model import PasteModel, parameter_count


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120_000)
    p.add_argument("--heldout", type=int, default=8_000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train, _ = build(args.n, args.seed, offline=args.offline)
    heldout, heldout_examples = build(args.heldout, args.seed + 1000, offline=args.offline)
    print(f"[train] {len(train)} train / {len(heldout)} held-out examples", file=sys.stderr)

    model = PasteModel(n_span_labels=len(LABELS), n_kinds=len(LEARNED_KINDS))
    print(f"[train] {parameter_count(model)} parameters", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    # Class weights: the learned kinds are roughly balanced by the generator; span labels are
    # dominated by O, so down-weight it a little.
    span_w = torch.ones(len(LABELS))
    span_w[0] = 0.5

    RUNS.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        model.quantize = epoch >= 1  # one warm-up epoch in float, then QAT
        losses = []
        for ids, mask, labels, kinds, _ in batches(train, args.batch, rng):
            ids_t = torch.from_numpy(ids)
            mask_t = torch.from_numpy(mask)
            span, kind = model(ids_t, mask_t)
            loss_span = F.cross_entropy(span.reshape(-1, span.shape[-1]), torch.from_numpy(labels).reshape(-1), weight=span_w, ignore_index=-100)
            kinds_t = torch.from_numpy(kinds)
            loss_kind = F.cross_entropy(kind, kinds_t, ignore_index=-1) if (kinds_t >= 0).any() else span.sum() * 0
            loss = loss_span + loss_kind
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 200 == 0:
                print(f"[train] epoch {epoch} step {step}/{total} loss {np.mean(losses[-200:]):.4f} ({time.time() - t0:.0f}s)", file=sys.stderr)
        model.quantize = True
        metrics = evaluate_model(model, heldout, heldout_examples)
        score = metrics["kind_accuracy"] + metrics["span_f1_micro"]
        print(f"[train] epoch {epoch}: loss {np.mean(losses):.4f} kind acc {metrics['kind_accuracy']:.4f} span F1 {metrics['span_f1_micro']:.4f} ({time.time() - t0:.0f}s)", file=sys.stderr)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": args.seed, "metrics": metrics, "labels": LABELS, "kinds": LEARNED_KINDS, "span_kinds": SPAN_KINDS}, RUNS / "best.pt")
    (RUNS / "metrics.json").write_text(json.dumps({"history": history, "parameters": parameter_count(model), "seconds": time.time() - t0}, indent=2))
    return history[-1]


if __name__ == "__main__":
    main()
