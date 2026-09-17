"""Export runs/latest.pt to ../model/{manifest.json,weights.txt,fixtures.json}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.quant import export

from .data import LABELS, generate, read_markup_file, DATA_DIR
from .features import EMBED_ROWS, FEATURE_COUNT, FEATURE_SIZES, featurize
from .gen import KINDS
from .model import LogTagger, dequantized, forward_numpy

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def main() -> None:
    torch.set_num_threads(2)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "runs" / "latest.pt"
    ck = torch.load(path, map_location="cpu")
    model = LogTagger()
    model.load_state_dict(ck["state"])
    tensors = model.tensors()
    manifest = export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-log",
            "hidden": model.hidden,
            "embedDim": model.embed_dim,
            "blocks": len(model.blocks),
            "featureCount": FEATURE_COUNT,
            "featureSizes": [list(x) for x in FEATURE_SIZES],
            "embedRows": EMBED_ROWS,
            "labels": LABELS,
            "kinds": KINDS,
            "checkpoint": {"seed": ck.get("seed"), "epochs": ck.get("epochs"), "lines": ck.get("lines")},
        },
    )
    print(f"exported {manifest['parameters']:,} parameters to {MODEL_DIR}")

    # Parity fixtures: computed from the dequantized tensors with the numpy reference forward.
    deq = dequantized(tensors)
    cases = []
    samples = [ex.text for ex in generate(60, 99)]
    unf = DATA_DIR / "unfamiliar.txt"
    if unf.exists():
        samples += [ex.text for ex in read_markup_file(unf)[:12]]
    picked = [s for s in samples if 0 < len(s) < 400][:32]
    picked += ["", "x", "   ", "🚀 café 日本語 emoji line", "\t\tat a.b(C.java:1)"]
    for text in picked:
        tokens, rows = featurize(text)
        feats = np.asarray(rows, dtype=np.int64).reshape(-1, FEATURE_COUNT)
        logits = forward_numpy(deq, feats, blocks=len(model.blocks)) if len(rows) else np.zeros((0, len(LABELS) + len(KINDS)), np.float32)
        cases.append({"text": text, "features": rows, "logits": [[round(float(v), 6) for v in row] for row in logits]})
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(cases) + "\n")

    # Sanity: torch (fake-quant) vs numpy dequantized forward on one case.
    model.set_quant(True)
    model.eval()
    text = picked[0]
    _, rows = featurize(text)
    with torch.no_grad():
        f = torch.from_numpy(np.asarray(rows, dtype=np.int64))[None]
        tags, _ = model(f, torch.ones(1, len(rows)))
    ref = forward_numpy(deq, np.asarray(rows), blocks=len(model.blocks))[:, : len(LABELS)]
    print(f"torch vs numpy max abs diff: {float(np.abs(tags[0].numpy() - ref).max()):.2e}; fixtures: {len(cases)}")


if __name__ == "__main__":
    main()
