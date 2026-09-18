"""A small, generic QAT training loop: AdamW + linear warm-up + cosine decay, fake
quantization from ``qat_from``, evaluation of the *quantized* model every epoch,
best-checkpoint selection, history.json and an optional wall-clock budget.

    history = train(
        model, make_batches, evaluate, loss=loss_fn,
        epochs=8, lr=3e-3, weight_decay=1e-4, qat_from=1, threads=2, seed=0,
        out_dir=run_dir, select="exact_match", minutes=18,
    )

* ``make_batches(epoch, rng)`` returns an iterable of batches (any object).
* ``loss(model, batch)`` returns a scalar tensor.
* ``evaluate(model)`` returns a flat ``dict[str, float]``; ``select`` names the key to
  maximise (prefix with ``-`` to minimise, e.g. ``"-loss"``).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .qat import set_quant

Batches = Callable[[int, np.random.Generator], Iterable[Any]]


def _checkpoint(model: nn.Module, epoch: int, seed: int, metrics: dict[str, float]) -> dict[str, Any]:
    ckpt: dict[str, Any] = {"state": model.state_dict(), "epoch": epoch, "seed": seed, "metrics": metrics}
    if hasattr(model, "config"):
        ckpt["config"] = model.config()
    return ckpt


def load_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load ``best.pt``/``last.pt`` into ``model`` (returns the checkpoint metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    return ckpt


def train(
    model: nn.Module,
    make_batches: Batches,
    evaluate: Callable[[nn.Module], dict[str, float]],
    *,
    loss: Callable[[nn.Module, Any], Tensor],
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    qat_from: int = 1,
    threads: int = 2,
    seed: int = 0,
    out_dir: str | Path,
    select: str,
    minutes: float | None = None,
    warmup: int = 300,
    clip: float = 1.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    maximise = not select.startswith("-")
    key = select.lstrip("-")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total: dict[str, int | None] = {"steps": None}

    def schedule(step: int) -> float:
        warm = min(1.0, (step + 1) / max(1, warmup))
        if total["steps"] is None:
            return warm
        progress = min(step, total["steps"]) / max(1, total["steps"])
        return warm * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    history: list[dict[str, Any]] = []
    best_value = -math.inf
    best: dict[str, Any] | None = None
    step = 0
    started = time.time()
    deadline = started + minutes * 60 if minutes is not None else None
    stop = False

    for epoch in range(epochs):
        set_quant(model, epoch >= qat_from)
        model.train()
        batches = make_batches(epoch, rng)
        if total["steps"] is None and hasattr(batches, "__len__"):
            total["steps"] = len(batches) * epochs  # type: ignore[arg-type]
        losses: list[float] = []
        epoch_steps = 0
        for batch in batches:
            value = loss(model, batch)
            opt.zero_grad(set_to_none=True)
            value.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            step += 1
            epoch_steps += 1
            losses.append(float(value.detach()))
            if step % 100 == 0:
                log(f"epoch {epoch} step {step} loss {np.mean(losses[-100:]):.4f} lr {sched.get_last_lr()[0]:.2e} qat={epoch >= qat_from} {time.time() - started:.0f}s")
            if deadline is not None and time.time() > deadline:
                log("wall-clock budget reached")
                stop = True
                break
        if total["steps"] is None:
            total["steps"] = epoch_steps * epochs

        set_quant(model, True)  # always evaluate what will ship
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model)
        row = {"epoch": epoch, "steps": step, "loss": float(np.mean(losses)) if losses else None, "elapsed_s": round(time.time() - started, 1), **metrics}
        history.append(row)
        log(json.dumps(row))
        value = metrics[key] if maximise else -metrics[key]
        if value >= best_value:
            best_value = value
            best = {"epoch": epoch, **metrics}
            torch.save(_checkpoint(model, epoch, seed, metrics), out / "best.pt")
        torch.save(_checkpoint(model, epoch, seed, metrics), out / "last.pt")
        (out / "history.json").write_text(json.dumps({"select": select, "best": best, "history": history}, indent=2) + "\n")
        if stop:
            break
    return {"select": select, "best": best, "history": history, "minutes": round((time.time() - started) / 60, 2)}
