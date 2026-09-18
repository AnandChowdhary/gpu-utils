"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

    uv run python -m gpu_paste.export [--run default]

fixtures.json uses the canonical format ({"cases": [{input, rows, logits, pooled}]}):
`logits` is the BIO span head, `pooled` the kind head. Both are computed by the family
model from the *decoded* int6 weights (exactly the float32 values the runtime loads), so
test/parity.test.ts and the WGSL harness can require a 1e-4 match.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS
from gpu_paste.dataset import RUNS
from gpu_paste.features import FEATURE_COUNT, featurize_tokens
from gpu_paste.model import build
from gpu_utils_training.export import export_package
from gpu_utils_training.features import tokenize
from gpu_utils_training.loop import load_checkpoint

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    args = ap.parse_args()
    model = build()
    ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
    metrics = ckpt.get("metrics") or {}
    fixtures = []
    for text in FIXTURE_TEXTS:
        tokens = tokenize(text)
        fixtures.append({"input": text, "rows": featurize_tokens(tokens), "tokens": [t.text for t in tokens]})
    manifest = export_package(
        model,
        MODEL_DIR,
        LABELS,
        {
            "name": "gpu-paste",
            "kinds": LEARNED_KINDS,
            "spanKinds": SPAN_KINDS,
            "checkpoint": {
                "run": args.run,
                "seed": ckpt.get("seed"),
                "epoch": ckpt.get("epoch"),
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "heldout": {k: metrics[k] for k in ("kind_accuracy", "span_f1_micro") if k in metrics},
            },
        },
        fixtures,
        slots=FEATURE_COUNT,
    )
    print(f"exported {manifest['parameters']:,} parameters and {len(fixtures)} fixtures to {MODEL_DIR}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tensors"}, indent=1))


if __name__ == "__main__":
    main()
