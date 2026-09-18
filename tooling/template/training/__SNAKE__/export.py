"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m __SNAKE__.export [--run default] [--random]

fixtures.json uses the canonical format ({"cases": [{input, rows, logits, pooled}]}) and
is computed from the decoded int6 weights, so test/parity.test.ts and the WGSL harness
compare against exactly what the runtime loads. `--random` exports an untrained model
(useful to bootstrap the package before the first training run).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch
from gpu_utils_training.export import export_package
from gpu_utils_training.features import tokenize
from gpu_utils_training.loop import load_checkpoint

from .data import generate
from .features import LABELS, SLOTS, featurize_tokens
from .model import build

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODEL_DIR = HERE.parent.parent / "model"

HAND_WRITTEN = [
    "call Ada Lovelace tomorrow",
    "email Grace about the meeting",
    "",
    "Meet Linus, Alan and Margaret at 5",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--random", action="store_true", help="export an untrained model")
    args = ap.parse_args()
    torch.manual_seed(0)
    model = build()
    checkpoint: dict[str, object] = {"run": None, "date": date.today().isoformat()}
    if not args.random:
        ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
        checkpoint = {"run": args.run, "epoch": ckpt.get("epoch"), "seed": ckpt.get("seed"), "metrics": ckpt.get("metrics"), "date": date.today().isoformat()}
    texts = HAND_WRITTEN + [e.text for e in generate(20, seed=99)]
    fixtures = []
    for text in texts:
        tokens = tokenize(text)
        fixtures.append({"input": text, "rows": featurize_tokens(tokens), "tokens": [t.text for t in tokens]})
    manifest = export_package(model, MODEL_DIR, LABELS, {"name": "__NAME__", "checkpoint": checkpoint}, fixtures, slots=SLOTS)
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tensors"}, indent=1))


if __name__ == "__main__":
    main()
