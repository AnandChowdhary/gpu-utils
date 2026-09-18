"""Train gpu-cite with quantization-aware training. ``uv run python -m gpu_cite.train``.

Default config (seed 0, 5 epochs over 120K synthetic references) runs in under 20 minutes
on two CPU threads and writes ``runs/<name>/best.pt`` plus ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gpu_utils_training.features import tokenize
from torch import Tensor

from .data import CACHE, read_jsonl
from .features import WIDTH, featurize_tokens
from .labels import ROLES, TAGS, TYPES
from .metrics import SpanScorer, entity_spans, field_spans, format_table
from .model import CiteTagger, constrained_transitions, count_params, crf_nll

RUNS = Path(__file__).resolve().parents[1] / "runs"


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
        toks = tokenize(r["text"])
        feats = featurize_tokens(r["text"], toks)
        assert len(feats) == len(r["tags"]), (r["text"], len(feats), len(r["tags"]))
        r["rows"] = feats
        parts.append(np.asarray(feats, dtype=np.int32).reshape(-1, WIDTH))
        lengths.append(len(feats))
    np.savez_compressed(cache, rows=np.concatenate(parts), lengths=np.asarray(lengths))
    return rows


def batches(rows: list[dict[str, Any]], batch_size: int, rng: random.Random, shuffle: bool) -> list[list[dict[str, Any]]]:
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tags"]))
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    return [[rows[i] for i in c] for c in chunks]


def collate(batch: list[dict[str, Any]]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    T = max(len(r["tags"]) for r in batch)
    B = len(batch)
    rows = torch.zeros(B, T, WIDTH, dtype=torch.long)
    mask = torch.zeros(B, T)
    tags = torch.zeros(B, T, dtype=torch.long)
    parts = torch.zeros(B, T, dtype=torch.long)
    types = torch.tensor([r["type"] for r in batch], dtype=torch.long)
    for i, r in enumerate(batch):
        n = len(r["tags"])
        rows[i, :n] = torch.as_tensor(r["rows"], dtype=torch.long)
        mask[i, :n] = 1.0
        tags[i, :n] = torch.as_tensor(r["tags"])
        parts[i, :n] = torch.as_tensor(r["parts"])
    return rows, mask, tags, parts, types


def viterbi_batch(emissions: Tensor, trans: Tensor, mask: Tensor) -> list[list[int]]:
    """Batched Viterbi over [B,T,K] emissions with a prefix mask; start may not be I-X."""
    B, T, K = emissions.shape
    R = len(ROLES)
    score = emissions[:, 0].clone()
    score[:, 1 + R :] = -1e4
    back = torch.zeros(B, T, K, dtype=torch.long)
    for t in range(1, T):
        cand = score.unsqueeze(2) + trans.unsqueeze(0)  # [B, from, to]
        best, arg = cand.max(dim=1)
        m = mask[:, t].unsqueeze(-1)
        score = (best + emissions[:, t]) * m + score * (1.0 - m)
        back[:, t] = arg
    lengths = mask.sum(dim=1).long().tolist()
    last = score.argmax(dim=1).tolist()
    out: list[list[int]] = []
    for i in range(B):
        n = lengths[i]
        path = [last[i]]
        for t in range(n - 1, 0, -1):
            path.append(int(back[i, t, path[-1]]))
        path.reverse()
        out.append(path)
    return out


@torch.no_grad()
def evaluate_rows(model: CiteTagger, rows: list[dict[str, Any]], batch_size: int = 256) -> dict[str, Any]:
    model.eval()
    trans = constrained_transitions(model.transitions())
    scorer = SpanScorer()
    tok_correct = tok_total = 0
    for batch in batches(rows, batch_size, random.Random(0), shuffle=False):
        r, m, tg, pt, ty = collate(batch)
        tags_l, parts_l, ty_l = model(r, m)
        paths = viterbi_batch(tags_l, trans, m)
        pred_parts = parts_l.argmax(-1)
        pred_types = ty_l.argmax(-1).tolist()
        for i, ex in enumerate(batch):
            toks = tokenize(ex["text"])
            n = len(ex["tags"])
            gold, pred = ex["tags"], paths[i]
            tok_correct += sum(int(a == b) for a, b in zip(gold, pred, strict=True))
            tok_total += n
            g_auth = entity_spans(ex["text"], gold, toks, "AUTHOR")
            p_auth = entity_spans(ex["text"], pred, toks, "AUTHOR")
            # full-record exact match also requires the given/family split on every name token
            gp = [p for p, t in zip(ex["parts"], gold, strict=True) if t]
            pp = [int(pred_parts[i, k]) for k, t in enumerate(pred) if t]
            parts_ok = gold == pred and gp == pp
            scorer.add(
                field_spans(ex["text"], gold, toks),
                field_spans(ex["text"], pred, toks),
                ex["type"],
                pred_types[i],
                g_auth,
                p_auth,
                extra_ok=parts_ok,
            )
    summary = scorer.summary()
    summary["token_accuracy"] = round(tok_correct / max(1, tok_total), 4)
    model.train()
    return summary


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_rows = read_jsonl(CACHE / "train.jsonl.gz")
    held_rows = read_jsonl(CACHE / "heldout.jsonl.gz")
    if args.limit:
        train_rows = train_rows[: args.limit]
    t0 = time.time()
    train_rows = featurize_set(train_rows, "feats_train")
    held_rows = featurize_set(held_rows, "feats_heldout")
    print(f"featurized {len(train_rows)} train / {len(held_rows)} held-out in {time.time() - t0:.0f}s", flush=True)

    model = CiteTagger(embed=args.embed, hidden=args.hidden, head=args.head)
    print(f"parameters: {count_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=50)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best: dict[str, Any] = {"exact_match": -1.0}
    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.quant = epoch >= args.qat_from
        losses: list[float] = []
        for batch in batches(train_rows, args.batch, rng, shuffle=True):
            r, m, tg, pt, ty = collate(batch)
            tags_l, parts_l, ty_l = model(r, m)
            loss = crf_nll(tags_l, model.transitions(), tg, m)
            author_mask = (tg > 0).float() * m
            part_loss = (ce(parts_l.reshape(-1, parts_l.shape[-1]), pt.reshape(-1)).reshape(m.shape) * author_mask).sum() / author_mask.sum().clamp(min=1.0)
            type_loss = ce(ty_l, ty).mean()
            total_loss = loss + 0.5 * part_loss + 0.5 * type_loss
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(total_loss))
            if step % 100 == 0:
                el = time.time() - start
                print(f"epoch {epoch} step {step}/{total} loss {np.mean(losses[-100:]):.3f} crf {float(loss):.3f} qat={model.quant} {el:.0f}s", flush=True)
        model.quant = True  # always evaluate the quantized model
        metrics = evaluate_rows(model, held_rows)
        print(f"epoch {epoch}: held-out exact {metrics['exact_match']:.3f} micro-F1 {metrics['micro_f1']:.3f} type {metrics.get('type_accuracy', 0):.3f} tokacc {metrics['token_accuracy']:.3f} ({time.time() - start:.0f}s)", flush=True)
        if metrics["exact_match"] >= best["exact_match"]:
            best = {**metrics, "epoch": epoch}
            torch.save({"state": model.state_dict(), "config": {"embed": args.embed, "hidden": args.hidden, "head": args.head}, "epoch": epoch, "seed": args.seed}, run_dir / "best.pt")
    print(format_table(best))
    best["params"] = count_params(model)
    best["train_examples"] = len(train_rows)
    best["minutes"] = round((time.time() - start) / 60, 1)
    (run_dir / "metrics.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps({k: v for k, v in best.items() if k != "fields"}, indent=1))
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="default")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--embed", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--head", type=int, default=48)
    ap.add_argument("--qat-from", type=int, default=1, help="enable int6 fake-quant from this epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args(argv)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
