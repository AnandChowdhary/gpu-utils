"""Export the promoted checkpoint to ../model/{manifest.json,weights.txt,fixtures.json}.

  uv run python -m gpu_view.export [--run default]

fixtures.json holds >= 20 cases with inputs, feature rows and logits computed by a
NumPy float64 reference forward over the *dequantized int6 weights* (decoded from
weights.txt exactly as runtime/weights.ts does), so test/parity.test.ts can assert the
TypeScript CPU path matches at 1e-4.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.quant import ALPHABET, export as quant_export

from . import features
from .generate import ROLES, dataset
from .model import CONV, HEAD_GATE, HIDDEN, ViewTagger

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
MODEL_DIR = HERE.parent.parent / "model"
LOOKUP = {ch: i - 32 for i, ch in enumerate(ALPHABET)}


def decode_weights(encoded: str, manifest: dict) -> dict[str, np.ndarray]:
    """Decode exactly like runtime/weights.ts: int * scale in float64, stored as float32."""
    out = {}
    for t in manifest["tensors"]:
        q = np.array([LOOKUP[c] for c in encoded[t["offset"] : t["offset"] + t["length"]]], dtype=np.float64)
        out[t["name"]] = (q * t["scale"]).astype(np.float32).astype(np.float64).reshape(t["shape"])
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def forward_numpy(w: dict[str, np.ndarray], rows: list[list[int]]) -> np.ndarray:
    """Float64 reference forward for one sequence; returns [n, roles + 1] logits."""
    n = len(rows)
    if n == 0:
        return np.zeros((0, w["output_bias"].shape[0]))
    emb = np.stack([w["embedding"][r].sum(axis=0) for r in rows])  # [n, H]
    enc = np.tile(w["encoder_bias"], (n, 1))
    for k in range(CONV):
        shift = k - CONV // 2
        for t in range(n):
            s = t + shift
            if 0 <= s < n:
                enc[t] += w["convolution"][k] * emb[s]
    enc = np.tanh(enc)
    gate = sigmoid(enc @ w["gate_weight"].T + w["gate_bias"])
    cand = (1 - gate) * np.tanh(enc @ w["candidate_weight"].T + w["candidate_bias"])
    fwd = np.zeros_like(cand)
    state = np.zeros(HIDDEN)
    for t in range(n):
        state = gate[t] * state + cand[t]
        fwd[t] = state
    bwd = np.zeros_like(cand)
    state = np.zeros(HIDDEN)
    for t in range(n - 1, -1, -1):
        state = gate[t] * state + cand[t]
        bwd[t] = state
    combined = np.tanh(enc + np.concatenate([fwd, bwd], axis=1) @ w["combine_weight"].T + w["combine_bias"])
    pooled = combined.mean(axis=0)
    context = sigmoid(pooled @ w["global_weight"].T + w["global_bias"]) * pooled
    joined = np.concatenate([combined, np.tile(context, (n, 1))], axis=1)
    hg = sigmoid(joined @ w["head_gate_weight"].T + w["head_gate_bias"])
    hidden = np.tanh(np.concatenate([joined, hg], axis=1) @ w["head_hidden_weight"].T + w["head_hidden_bias"])
    return hidden @ w["output_weight"].T + w["output_bias"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    args = ap.parse_args()
    run_dir = RUNS / args.run
    model = ViewTagger(features.FEATURE_ROWS, len(ROLES))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}

    tensors = {name: getattr(model, name).detach().numpy() for name in ViewTagger.TENSOR_NAMES}
    manifest = quant_export(
        tensors,
        MODEL_DIR,
        {
            "name": "gpu-view",
            "hidden": HIDDEN,
            "headGate": HEAD_GATE,
            "conv": CONV,
            "labels": ROLES,
            "featureRows": features.FEATURE_ROWS,
            "slots": features.SLOTS,
            "blocks": [{"name": n, "offset": features.OFFSET[n], "size": s} for n, s in features.BLOCKS],
            "checkpoint": {"run": args.run, "seed": metrics.get("args", {}).get("seed"),
                           "epochs": metrics.get("args", {}).get("epochs"),
                           "samples": metrics.get("args", {}).get("samples"), "date": date.today().isoformat()},
        },
    )
    encoded = (MODEL_DIR / "weights.txt").read_text()
    w = decode_weights(encoded, manifest)

    # Fixtures: generated queries over training and held-out schemas plus a few hand-written ones.
    cases = []
    examples = dataset("train", 14, seed=31) + dataset("eval", 10, seed=32)
    hand = [
        ("total revenue by region this quarter, top 10, as a bar chart", {"fields": [
            {"name": "revenue", "kind": "number"}, {"name": "region", "kind": "enum", "values": ["EMEA", "APAC"]},
            {"name": "closed_at", "kind": "date"}]}),
        ("open issues assigned to me sorted by priority", {"fields": [
            {"name": "status", "kind": "enum", "values": ["open", "closed"]},
            {"name": "assignee", "kind": "text", "aliases": ["assigned"]},
            {"name": "priority", "kind": "enum", "values": ["low", "high"]}]}),
    ]
    # Cross-check the NumPy reference against torch *on the decoded int6 weights*: that is what
    # ships. (Re-running fake-quant in float64 flips weights that sit on rounding ties.)
    model.qat = False
    model.double()
    with torch.no_grad():
        for name in ViewTagger.TENSOR_NAMES:
            getattr(model, name).copy_(torch.from_numpy(w[name]))
    worst = 0.0
    for ex_text, ex_schema in [(e.text, e.schema) for e in examples] + hand:
        toks, rows = features.featurize(ex_text, ex_schema)
        logits = forward_numpy(w, rows)
        padded = np.full((1, len(rows), features.SLOTS), features.PADDING_ROW, dtype=np.int64)
        for i, r in enumerate(rows):
            padded[0, i, : len(r)] = r
        with torch.no_grad():
            tr, tb = model(torch.from_numpy(padded), torch.ones(1, len(rows), dtype=torch.bool))
        torch_logits = torch.cat([tr[0], tb[0].unsqueeze(-1)], dim=-1).numpy()
        worst = max(worst, float(np.abs(torch_logits - logits).max()))
        cases.append({"text": ex_text, "schema": ex_schema, "tokens": [t.text for t in toks], "rows": rows,
                      "logits": [[round(float(v), 6) for v in row] for row in logits]})
    (MODEL_DIR / "fixtures.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"exported {manifest['parameters']:,} parameters; numpy vs torch max |Δ| = {worst:.2e}; {len(cases)} fixtures")
    assert worst < 1e-4


if __name__ == "__main__":
    main()
