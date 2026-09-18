"""Evaluation: learned-kind accuracy + confusion matrix and per-span-kind F1.

    uv run python -m gpu_paste.evaluate            # held-out generated set + unfamiliar set

The unfamiliar set lives in ../test/unfamiliar.json and is shared with the TypeScript
end-to-end test; here only the learned parts (kind head on learned kinds, model span kinds)
are scored, the TypeScript test scores the whole rules+model pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS, Example
from gpu_paste.dataset import Encoded, batches, build, encode
from gpu_paste.model import PasteModel

UNFAMILIAR = Path(__file__).resolve().parents[2] / "test" / "unfamiliar.json"


def constrained_decode(logits: np.ndarray) -> list[int]:
    """Viterbi with BIO constraints (O/B-x → I-y forbidden unless x == y). Mirrors src/decode.ts."""
    n, k = logits.shape
    trans = np.zeros((k, k), dtype=np.float64)
    for to in range(1, k):
        if LABELS[to].startswith("I-"):
            kind = LABELS[to][2:]
            for frm in range(k):
                if LABELS[frm] in (f"B-{kind}", f"I-{kind}"):
                    continue
                trans[frm, to] = -np.inf
    score = logits[0].astype(np.float64).copy()
    for to in range(k):
        if LABELS[to].startswith("I-"):
            score[to] = -np.inf
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i] = cand.argmax(0)
        score = cand.max(0) + logits[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def spans_from_labels(path: list[int]) -> list[tuple[int, int, str]]:
    """Token-index spans [start, end) with kind."""
    out = []
    start = -1
    kind = ""
    for i, lab in enumerate(path + [0]):
        name = LABELS[lab] if lab < len(LABELS) else "O"
        if name.startswith("B-") or name == "O" or (name.startswith("I-") and name[2:] != kind):
            if start >= 0:
                out.append((start, i, kind))
                start = -1
            if name.startswith("B-") or name.startswith("I-"):
                start, kind = i, name[2:]
    return out


def gold_spans(labels: np.ndarray) -> list[tuple[int, int, str]]:
    return spans_from_labels([int(x) for x in labels])


def evaluate_model(model: PasteModel, encoded: list[Encoded], examples: list[Example] | None = None, batch_size: int = 128) -> dict:
    model.eval()
    k = len(LEARNED_KINDS)
    confusion = np.zeros((k, k), dtype=np.int64)
    tp = {s: 0 for s in SPAN_KINDS}
    fp = {s: 0 for s in SPAN_KINDS}
    fn = {s: 0 for s in SPAN_KINDS}
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for ids, mask, labels, kinds, idx in batches(encoded, batch_size, rng, shuffle=False):
            span, kind = model(torch.from_numpy(ids), torch.from_numpy(mask))
            span = span.numpy()
            pred_kind = kind.argmax(-1).numpy()
            for j, i in enumerate(idx):
                if kinds[j] >= 0:
                    confusion[kinds[j], pred_kind[j]] += 1
                n = int(mask[j].sum())
                pred = set(spans_from_labels(constrained_decode(span[j, :n])))
                gold = set(gold_spans(labels[j, :n]))
                for s in pred & gold:
                    tp[s[2]] += 1
                for s in pred - gold:
                    fp[s[2]] += 1
                for s in gold - pred:
                    fn[s[2]] += 1
    per_kind = {}
    for s in SPAN_KINDS:
        p = tp[s] / (tp[s] + fp[s]) if tp[s] + fp[s] else 0.0
        r = tp[s] / (tp[s] + fn[s]) if tp[s] + fn[s] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_kind[s] = {"precision": p, "recall": r, "f1": f1, "support": tp[s] + fn[s]}
    ttp, tfp, tfn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = ttp / (ttp + tfp) if ttp + tfp else 0.0
    mr = ttp / (ttp + tfn) if ttp + tfn else 0.0
    return {
        "kind_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "kind_support": int(confusion.sum()),
        "confusion": confusion.tolist(),
        "kinds": LEARNED_KINDS,
        "span_f1_micro": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        "span_precision_micro": mp,
        "span_recall_micro": mr,
        "spans": per_kind,
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


def main() -> None:
    from gpu_paste.dataset import RUNS

    ckpt = torch.load(RUNS / "best.pt", map_location="cpu")
    model = PasteModel(n_span_labels=len(LABELS), n_kinds=len(LEARNED_KINDS))
    model.load_state_dict(ckpt["state_dict"])
    model.quantize = True
    heldout, examples = build(8000, ckpt["seed"] + 1000)
    m = evaluate_model(model, heldout, examples)
    print("== held-out (generated) ==")
    print(f"kind accuracy {m['kind_accuracy']:.4f} on {m['kind_support']}; span micro-F1 {m['span_f1_micro']:.4f}")
    print(format_confusion(m))
    for s, v in m["spans"].items():
        print(f"  {s:<8} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")
    if UNFAMILIAR.exists():
        unf = load_unfamiliar()
        enc = [encode(e) for e in unf]
        u = evaluate_model(model, enc, unf)
        print("== unfamiliar (hand-written, model heads only) ==")
        print(f"kind accuracy {u['kind_accuracy']:.4f} on {u['kind_support']}; span micro-F1 {u['span_f1_micro']:.4f}")
        print(format_confusion(u))
        for s, v in u["spans"].items():
            print(f"  {s:<8} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")
        json.dump({"heldout": m, "unfamiliar": u}, open(RUNS / "eval.json", "w"), indent=2)
    else:
        json.dump({"heldout": m}, open(RUNS / "eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
