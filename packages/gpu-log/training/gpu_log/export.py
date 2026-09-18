"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_log.export [--run default] [--random]

fixtures.json uses the canonical format ({"cases": [{input, rows, logits, pooled}]}) computed
from the decoded int6 weights, so test/parity.test.ts and the WGSL harness compare against
exactly what the runtime loads.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch
from gpu_utils_training.export import export_package
from gpu_utils_training.loop import load_checkpoint

from .data import DATA_DIR, KINDS, LABELS, generate, read_markup_file
from .features import FEATURE_COUNT, FEATURE_SIZES, featurize
from .model import build

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODEL_DIR = HERE.parent.parent / "model"

HAND_WRITTEN = [
    "",
    "x",
    "   ",
    "🚀 café 日本語 emoji line",
    "\t\tat a.b(C.java:1)",
    "2024-01-15 10:30:00,123 [main] INFO  com.example.Foo - Started server on port 8080",
    "Jan 15 10:30:00 web-01 sshd[1234]: Accepted publickey for alice from 10.0.0.9 port 22 ssh2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--random", action="store_true", help="export an untrained model")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = build()
    checkpoint: dict[str, object] = {"run": None, "date": date.today().isoformat()}
    if not args.random:
        ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
        checkpoint = {"run": args.run, "epoch": ckpt.get("epoch"), "seed": ckpt.get("seed"), "metrics": ckpt.get("metrics"), "date": date.today().isoformat()}
    texts = HAND_WRITTEN + [e.text for e in generate(40, seed=99) if 0 < len(e.text) < 160][:18]
    unf = DATA_DIR / "unfamiliar_v2.txt"
    if unf.exists():
        texts += [e.text for e in read_markup_file(unf) if len(e.text) < 160][:6]
    fixtures = []
    for text in texts:
        tokens, rows = featurize(text)
        fixtures.append({"input": text, "rows": rows, "tokens": [t.text for t in tokens]})
    manifest = export_package(
        model,
        MODEL_DIR,
        LABELS,
        {"name": "gpu-log", "kinds": KINDS, "featureSizes": [list(x) for x in FEATURE_SIZES], "checkpoint": checkpoint},
        fixtures,
        slots=FEATURE_COUNT,
    )
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("tensors", "labels")}, indent=1))


if __name__ == "__main__":
    main()
