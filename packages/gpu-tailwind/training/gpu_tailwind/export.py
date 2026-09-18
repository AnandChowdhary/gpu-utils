"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_tailwind.export [--run default] [--random]

fixtures.json uses the canonical format ({"cases": [{input, rows, logits, pooled}]}) and
is computed from the decoded int6 weights, so test/parity.test.ts and the WGSL harness
compare against exactly what the runtime loads.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch
from gpu_utils_training.export import export_package
from gpu_utils_training.loop import load_checkpoint

from .batches import load
from .features import WIDTH, featurize
from .labels import LABELS
from .model import build

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODEL_DIR = HERE.parent.parent / "model"
EVAL = HERE.parent.parent / "eval"


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
    texts = [c["text"] for c in json.loads((EVAL / "unfamiliar-v1.json").read_text())[:10]]
    texts += [c["text"] for c in json.loads((EVAL / "unfamiliar-v2.json").read_text())[:10]]
    texts += [r["text"] for r in load("heldout", 8)]
    texts.append("")
    fixtures = [{"input": t, "rows": featurize(t)} for t in texts]
    manifest = export_package(model, MODEL_DIR, LABELS, {"name": "gpu-tailwind", "checkpoint": checkpoint}, fixtures, slots=WIDTH)
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")


if __name__ == "__main__":
    main()
