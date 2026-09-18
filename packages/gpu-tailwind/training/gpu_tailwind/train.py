"""Train gpu-tailwind with int6 quantization-aware training.

    uv run python -m gpu_tailwind.train [--minutes 16] [--seed 0] [--epochs 8]

Writes runs/best.pt and runs/metrics.json. CPU only, 2 threads (shared box).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import Tensor

from .data import CACHE, LABELS
from .features import WIDTH, featurize
from .model import OUT, Tagger, count_params

RUNS = Path(__file__).resolve().parents[1] / "runs"
LABEL_ID = {l: i for i, l in enumerate(LABELS)}


def load(name: str, limit: int | None = None) -> list[dict]:
    path = CACHE / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `uv run python -m gpu_tailwind.data` first")
    rows = [json.loads(l) for l in path.open()]
    return rows[:limit] if limit else rows


def encode(rows: list[dict]) -> list[tuple[list[list[int]], list[int], list[int]]]:
    out = []
    for r in rows:
        feats = featurize(r["text"])
        assert len(feats) == len(r["labels"]), r["text"]
        out.append((feats, [LABEL_ID[l] for l in r["labels"]], r["boundary"]))
    return out


def batches(data: list, bsz: int, rng: random.Random, shuffle: bool):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
        # bucket by length for speed
        chunks = [idx[i : i + bsz * 50] for i in range(0, len(idx), bsz * 50)]
        idx = [j for c in chunks for j in sorted(c, key=lambda k: len(data[k][1]))]
    groups = [idx[i : i + bsz] for i in range(0, len(idx), bsz)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        t = max(len(data[j][1]) for j in g)
        feats = torch.zeros(len(g), t, WIDTH, dtype=torch.long)
        labels = torch.full((len(g), t), -100, dtype=torch.long)
        bound = torch.zeros(len(g), t)
        mask = torch.zeros(len(g), t, dtype=torch.bool)
        for i, j in enumerate(g):
            f, l, b = data[j]
            n = len(l)
            feats[i, :n] = torch.tensor(f)
            labels[i, :n] = torch.tensor(l)
            bound[i, :n] = torch.tensor(b, dtype=torch.float)
            mask[i, :n] = True
        yield feats, labels, bound, mask


def loss_fn(out: Tensor, labels: Tensor, bound: Tensor, mask: Tensor) -> Tensor:
    roles = out[..., : OUT - 1]
    ce = torch.nn.functional.cross_entropy(roles.reshape(-1, OUT - 1), labels.reshape(-1), ignore_index=-100)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[..., OUT - 1], bound, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp(min=1)
    return ce + 0.5 * bce


@torch.no_grad()
def evaluate(model: Tagger, data: list, bsz: int = 256) -> dict[str, float]:
    model.eval()
    correct = total = 0
    seq_ok = seq_n = 0
    b_correct = b_total = 0
    for feats, labels, bound, mask in batches(data, bsz, random.Random(0), False):
        out = model(feats, mask)
        pred = out[..., : OUT - 1].argmax(-1)
        ok = (pred == labels) | ~mask
        correct += int(((pred == labels) & mask).sum())
        total += int(mask.sum())
        seq_ok += int(ok.all(1).sum())
        seq_n += feats.shape[0]
        bp = (out[..., OUT - 1] > 0).float()
        b_correct += int(((bp == bound) & mask).sum())
        b_total += int(mask.sum())
    model.train()
    return {"token_acc": correct / max(total, 1), "seq_acc": seq_ok / max(seq_n, 1), "boundary_acc": b_correct / max(b_total, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    RUNS.mkdir(exist_ok=True)

    train = encode(load("train", args.limit))
    heldout = encode(load("heldout", 3000))
    model = Tagger()
    print(f"params: {count_params(model)}  train: {len(train)}  heldout: {len(heldout)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train) / args.bsz)
    total_steps = steps_per_epoch * args.epochs
    warm = min(300, total_steps // 20)

    def lr_at(step: int) -> float:
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 1e-4 + (args.lr - 1e-4) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    start = time.time()
    step = 0
    best = -1.0
    metrics: dict = {}
    stop = False
    for epoch in range(args.epochs):
        model.quant = epoch >= 1
        running = 0.0
        n = 0
        for feats, labels, bound, mask in batches(train, args.bsz, rng, True):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            out = model(feats, mask)
            loss = loss_fn(out, labels, bound, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss)
            n += 1
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / n:.4f} lr {lr_at(step):.2e} {time.time() - start:.0f}s", flush=True)
            if (time.time() - start) / 60 > args.minutes:
                stop = True
                break
        model.quant = True
        ev = evaluate(model, heldout)
        print(f"epoch {epoch} done: loss {running / max(n, 1):.4f} heldout {ev} ({time.time() - start:.0f}s)", flush=True)
        if ev["token_acc"] > best:
            best = ev["token_acc"]
            metrics = {"epoch": epoch, "step": step, "seed": args.seed, **ev, "params": count_params(model), "minutes": (time.time() - start) / 60}
            torch.save(model.state_dict(), RUNS / "best.pt")
            (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if stop:
            print("time budget reached")
            break
    print("best:", metrics)


if __name__ == "__main__":
    main()
