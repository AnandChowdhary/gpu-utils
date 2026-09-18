"""Export runs/best.pt to ../model/{manifest.json,weights.txt,fixtures.json}.

uv run python -m gpu_tailwind.export
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.quant import export, quantize

from .data import CACHE, LABELS
from .features import ROWS, WIDTH, featurize
from .model import OUT, D, H, Tagger
from .train import RUNS

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
UNFAMILIAR = Path(__file__).resolve().parents[1] / "data" / "unfamiliar.json"
ORDER = [
    "emb",
    "wa_f",
    "ba_f",
    "wu_f",
    "bu_f",
    "wa_b",
    "ba_b",
    "wu_b",
    "bu_b",
    "conv",
    "bc",
    "w1",
    "wg",
    "b1",
    "w2",
    "b2",
]


def dequantized(model: Tagger) -> Tagger:
    """Model whose float weights equal exactly what the runtime decodes from weights.txt."""
    out = Tagger()
    with torch.no_grad():
        for name, p in model.named_parameters():
            q, scale = quantize(p.detach().numpy().astype(np.float32))
            getattr(out, name).copy_(
                torch.from_numpy(
                    (q.astype(np.float32) * np.float32(scale)).astype(np.float32)
                )
            )
    out.quant = False
    out.eval()
    return out


def logits_for(model: Tagger, text: str) -> tuple[list[list[int]], list[float]]:
    rows = featurize(text)
    feats = torch.tensor([rows], dtype=torch.long)
    mask = torch.ones(1, len(rows), dtype=torch.bool)
    with torch.no_grad():
        out = model(feats, mask)[0]
    return rows, [round(float(x), 6) for x in out.reshape(-1)]


def main() -> None:
    model = Tagger()
    model.load_state_dict(torch.load(RUNS / "best.pt"))
    metrics = json.loads((RUNS / "metrics.json").read_text())
    tensors = {n: model.state_dict()[n].numpy().astype(np.float32) for n in ORDER}
    manifest = export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-tailwind",
            "hidden": D,
            "head": H,
            "out": OUT,
            "rows": ROWS,
            "width": WIDTH,
            "labels": LABELS,
            "checkpoint": {
                "seed": metrics.get("seed"),
                "epoch": metrics.get("epoch"),
                "step": metrics.get("step"),
                "token_acc": metrics.get("token_acc"),
                "seq_acc": metrics.get("seq_acc"),
            },
        },
    )
    dq = dequantized(model)
    texts = [r["text"] for r in json.loads(UNFAMILIAR.read_text())[:12]]
    with (CACHE / "heldout.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= 14:
                break
            texts.append(json.loads(line)["text"])
    fixtures = []
    for t in texts:
        rows, logits = logits_for(dq, t)
        fixtures.append({"text": t, "features": rows, "logits": logits})
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(fixtures) + "\n")
    print(
        f"exported {manifest['parameters']} params, {len(fixtures)} fixtures -> {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
