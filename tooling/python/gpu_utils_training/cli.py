"""Shared argparse defaults so a package's train.py is ~20 lines.

    from gpu_utils_training.cli import training_parser, run_dir
    args = training_parser("Train gpu-example", epochs=8).parse_args()
    out = run_dir(__file__, args)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "seed": 0,
    "threads": 2,
    "minutes": None,
    "run": "default",
    "lr": 3e-3,
    "batch": 128,
    "weight_decay": 1e-4,
    "qat_from": 1,
    "warmup": 300,
}


def training_parser(description: str = "", **defaults: Any) -> argparse.ArgumentParser:
    """Parser with --epochs --seed --threads --minutes --run --lr --batch --weight-decay --qat-from --warmup."""
    d = {**DEFAULTS, **defaults}
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--threads", type=int, default=d["threads"], help="torch threads (the CI box is shared)")
    ap.add_argument("--minutes", type=float, default=d["minutes"], help="wall-clock budget; stops early and keeps the best checkpoint")
    ap.add_argument("--run", default=d["run"], help="run name under training/runs/")
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--batch", type=int, default=d["batch"])
    ap.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    ap.add_argument("--qat-from", type=int, default=d["qat_from"], help="epoch index from which int6 fake-quant is on")
    ap.add_argument("--warmup", type=int, default=d["warmup"], help="linear warm-up steps before the cosine decay")
    return ap


def run_dir(anchor: str | Path, args: argparse.Namespace) -> Path:
    """``training/runs/<run>`` next to the package module that calls this (``anchor=__file__``)."""
    out = Path(anchor).resolve().parent.parent / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    return out


def loop_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The subset of parsed args that loop.train accepts as keyword arguments."""
    return {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "qat_from": args.qat_from,
        "threads": args.threads,
        "seed": args.seed,
        "minutes": args.minutes,
        "warmup": args.warmup,
    }
