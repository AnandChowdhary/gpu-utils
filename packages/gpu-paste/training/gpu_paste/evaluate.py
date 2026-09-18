"""Evaluation: learned-kind accuracy + confusion matrix and per-span-kind F1.

    uv run python -m gpu_paste.evaluate [--run default]   # held-out generated set + unfamiliar set

Decoding uses the shared Viterbi/BIO tables (gpu_utils_training.decode, fixture-tested
against runtime decode.ts) and the shared span metrics (gpu_utils_training.metrics).
The unfamiliar set lives in ../test/unfamiliar.json and is shared with the TypeScript
end-to-end test; here only the learned parts (kind head on learned kinds, model span kinds)
are scored, the TypeScript test scores the whole rules+model pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS, Example
from gpu_paste.dataset import RUNS, Batches, Encoded, build, encode
from gpu_paste.model import build as build_model
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.loop import load_checkpoint
from gpu_utils_training.metrics import bio_to_spans, confusion, span_prf

UNFAMILIAR = Path(__file__).resolve().parents[2] / "test" / "unfamiliar.json"
TRANSITIONS = bio_transitions(LABELS)
START = bio_start_mask(LABELS)
EMPTY = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


@torch.no_grad()
def evaluate_model(model: nn.Module, encoded: list[Encoded], batch_size: int = 128) -> dict:
    """Kind accuracy/confusion over examples with a learned kind, exact-span P/R/F1 over all."""
    model.eval()
    padding_id: int = model.padding_id  # type: ignore[attr-defined]
    pred_spans, gold_spans = [], []
    kind_pred: list[int] = []
    kind_gold: list[int] = []
    for rows, mask, labels, kinds, _ in Batches(encoded, batch_size, None, padding_id, shuffle=False):
        out = model(rows, mask)
        tags = out["tags"].numpy()
        pooled = out["pooled"]
        assert pooled is not None
        best_kind = pooled.argmax(-1).numpy()
        lengths = mask.sum(dim=1).numpy()
        for j in range(rows.shape[0]):
            if int(kinds[j]) >= 0:
                kind_gold.append(int(kinds[j]))
                kind_pred.append(int(best_kind[j]))
            n = int(lengths[j])
            em = tags[j, :n].astype(np.float64)
            if n:
                em[0] += START
            path = viterbi(em, TRANSITIONS)
            pred_spans.append(bio_to_spans([LABELS[i] for i in path]))
            gold_spans.append(bio_to_spans([LABELS[int(i)] for i in labels[j, :n]]))
    prf = span_prf(pred_spans, gold_spans)
    per_label = prf["per_label"]
    micro = prf["micro"]
    assert isinstance(per_label, dict) and isinstance(micro, dict)
    cm = confusion(np.asarray(kind_pred, dtype=np.int64), np.asarray(kind_gold, dtype=np.int64), len(LEARNED_KINDS))
    return {
        "kind_accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "kind_support": int(cm.sum()),
        "confusion": cm.tolist(),
        "kinds": LEARNED_KINDS,
        "span_f1_micro": float(micro["f1"]),
        "span_precision_micro": float(micro["precision"]),
        "span_recall_micro": float(micro["recall"]),
        "spans": {s: dict(per_label.get(s, EMPTY)) for s in SPAN_KINDS},
    }


def summary(m: dict) -> dict[str, float]:
    """The flat subset loop.train logs and selects on (score = kind accuracy + span F1)."""
    return {
        "score": m["kind_accuracy"] + m["span_f1_micro"],
        "kind_accuracy": m["kind_accuracy"],
        "span_f1_micro": m["span_f1_micro"],
        "span_precision_micro": m["span_precision_micro"],
        "span_recall_micro": m["span_recall_micro"],
    }


def load_unfamiliar() -> list[Example]:
    """Cases from test/unfamiliar.json whose expected kind/spans exercise the learned parts."""
    cases = json.loads(UNFAMILIAR.read_text())
    out: list[Example] = []
    for c in cases:
        text = c["text"]
        spans = []
        for s in c.get("spans", []):
            if s["kind"] not in SPAN_KINDS:
                continue
            start = text.find(s["text"])
            assert start >= 0, (s, text)
            s16 = len(text[:start].encode("utf-16-le")) // 2
            e16 = s16 + len(s["text"].encode("utf-16-le")) // 2
            spans.append((s16, e16, s["kind"]))
        kind = c["kind"] if c["kind"] in LEARNED_KINDS else None
        out.append(Example(text, kind, spans))
    return out


def format_confusion(m: dict) -> str:
    kinds = m["kinds"]
    w = max(len(k) for k in kinds) + 1
    lines = [" " * w + " ".join(f"{k[:8]:>8}" for k in kinds)]
    for i, k in enumerate(kinds):
        lines.append(f"{k:<{w}}" + " ".join(f"{m['confusion'][i][j]:>8}" for j in range(len(kinds))))
    return "\n".join(lines)


def report(title: str, m: dict) -> None:
    print(f"== {title} ==")
    print(f"kind accuracy {m['kind_accuracy']:.4f} on {m['kind_support']}; span micro-F1 {m['span_f1_micro']:.4f} (P {m['span_precision_micro']:.3f} R {m['span_recall_micro']:.3f})")
    print(format_confusion(m))
    for s, v in m["spans"].items():
        print(f"  {s:<8} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--heldout", type=int, default=8_000)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = build_model()
    ckpt = load_checkpoint(model, RUNS / args.run / "best.pt")
    model.quant = True
    heldout, _ = build(args.heldout, int(ckpt["seed"]) + 1000)
    results = {"heldout": evaluate_model(model, heldout)}
    report("held-out (generated)", results["heldout"])
    if UNFAMILIAR.exists():
        unf = load_unfamiliar()
        results["unfamiliar"] = evaluate_model(model, [encode(e) for e in unf])
        report("unfamiliar (hand-written, model heads only)", results["unfamiliar"])
    (RUNS / args.run / "eval.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
