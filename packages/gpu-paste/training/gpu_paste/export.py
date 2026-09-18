"""Export runs/best.pt to ../model/{manifest.json,weights.txt,fixtures.json}.

Fixtures are computed with the *dequantized* int6 weights (exactly the float32 values the
runtime decodes), so test/parity.test.ts can require CPU logits to match at 1e-4.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS
from gpu_paste.dataset import RUNS
from gpu_paste.features import FEATURE_COUNT, TOTAL_ROWS, featurize
from gpu_paste.model import DIM, KIND_HIDDEN, MIX, PasteModel
from gpu_utils_training.features import tokenize
from gpu_utils_training.quant import export, quantize

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"

FIXTURE_TEXTS = [
    "Jane Doe\nSenior Engineer, Acme Corp\n+1 (555) 123-4567\njane@acme.com",
    "221B Baker Street\nLondon NW1 6XE\nUnited Kingdom",
    "Please wire €1.234,56 to Müller GmbH before 5 March 2025.",
    "- milk\n- eggs\n- bread",
    "def add(a, b):\n    return a + b",
    "# Title\n\nSome *text* with a [link](https://example.com).",
    "Call me tomorrow at 3pm, ask for Dr. Ahmed Khan.",
    "Hauptstraße 12, 10115 Berlin",
    "ok sounds good",
    "Invoice #4521 — Northwind Traders — $12,400.00 due Friday",
    "김민준\n서울특별시 강남구 테헤란로 152",
    "for (let i = 0; i < 10; i++) console.log(i);",
    "Meeting with Priya Sharma (Infosys) on 2024-11-02.",
    "1. Wake up\n2. Coffee\n3. Ship it",
    "Rechnung über CHF 1'250.00, fällig am 12. März.",
    "Tel: 020 7946 0958\nEmail: sam@example.org",
    "The quick brown fox jumps over the lazy dog.",
    "SELECT * FROM users WHERE id = 42;",
    "Rua das Flores 123, 01310-100 São Paulo, Brasil",
    "Thanks!\n\n-- \nMaria López\nHead of Design | Nova Studio\nM: +34 612 345 678",
    "> quoted reply\n\nSee you Thursday morning.",
    "apples, bananas, cherries, dates",
    "Vertex Capital AG raised ¥9,070,124 last year.",
    "😀 hello   world\t\ttabs",
    "",
    "x",
]


def dequantized_tensors(model: PasteModel) -> dict[str, np.ndarray]:
    """Per-tensor int6 round trip computed the same way runtime/weights.ts decodes."""
    out = {}
    for name, t in model.export_tensors().items():
        arr = t.numpy().astype(np.float32)
        q, scale = quantize(arr)
        out[name] = (q.astype(np.float64) * scale).astype(np.float32).reshape(arr.shape)
    return out


def load_dequantized(model: PasteModel) -> PasteModel:
    deq = dequantized_tensors(model)
    m = PasteModel(n_span_labels=len(LABELS), n_kinds=len(LEARNED_KINDS))
    m.quantize = False
    mapping = {
        "embed": m.embed.weight, "gate_f.w": m.gate_f.weight, "gate_f.b": m.gate_f.bias,
        "gate_b.w": m.gate_b.weight, "gate_b.b": m.gate_b.bias, "mix.w": m.mix.weight, "mix.b": m.mix.bias,
        "head.w": m.head.weight, "head.b": m.head.bias, "out.w": m.out.weight, "out.b": m.out.bias,
        "kind1.w": m.kind1.weight, "kind1.b": m.kind1.bias, "kind2.w": m.kind2.weight, "kind2.b": m.kind2.bias,
    }
    with torch.no_grad():
        for name, param in mapping.items():
            param.copy_(torch.from_numpy(deq[name]))
    m.eval()
    return m


def forward_text(m: PasteModel, text: str) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    rows = featurize(text)
    if not rows:
        return rows, np.zeros((0, len(LABELS)), dtype=np.float32), None
    ids = torch.tensor(rows, dtype=torch.int64).unsqueeze(0)
    mask = torch.ones(1, len(rows), dtype=torch.bool)
    with torch.no_grad():
        span, kind = m(ids, mask)
    return rows, span[0].numpy(), kind[0].numpy()


def main() -> None:
    ckpt = torch.load(RUNS / "best.pt", map_location="cpu")
    model = PasteModel(n_span_labels=len(LABELS), n_kinds=len(LEARNED_KINDS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tensors = {k: v.numpy().astype(np.float32) for k, v in model.export_tensors().items()}
    manifest = export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-paste",
            "dim": DIM,
            "mix": MIX,
            "kindHidden": KIND_HIDDEN,
            "featureCount": FEATURE_COUNT,
            "rows": TOTAL_ROWS,
            "labels": LABELS,
            "kinds": LEARNED_KINDS,
            "spanKinds": SPAN_KINDS,
            "checkpoint": {
                "seed": ckpt["seed"],
                "epoch": ckpt["epoch"],
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "heldout": {"kind_accuracy": ckpt["metrics"]["kind_accuracy"], "span_f1_micro": ckpt["metrics"]["span_f1_micro"]},
            },
        },
    )
    deq = load_dequantized(model)
    fixtures = []
    for text in FIXTURE_TEXTS:
        rows, span, kind = forward_text(deq, text)
        fixtures.append(
            {
                "text": text,
                "tokens": [t.text for t in tokenize(text)],
                "features": rows,
                "span": [round(float(v), 6) for v in span.ravel()],
                "kind": [round(float(v), 6) for v in kind] if kind is not None else [],
            }
        )
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False) + "\n")
    print(f"exported {manifest['parameters']} parameters, {len(fixtures)} fixtures → {MODEL_DIR}")


if __name__ == "__main__":
    main()
