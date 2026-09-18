"""Tagging metrics: BIO spans, span P/R/F1 (micro, macro, per label), exact match,
token accuracy and confusion matrices. Pure Python/NumPy so evaluate.py stays tiny."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

Span = tuple[str, int, int]  # (label, start token, end token) half-open


def bio_to_spans(tags: Sequence[str]) -> list[Span]:
    """Decode a BIO tag sequence into ``(label, start, end)`` token spans.

    A stray ``I-X`` (after ``O`` or a different label) starts a new span, which is the
    lenient convention used by conlleval and by runtime/bio.ts.
    """
    spans: list[Span] = []
    current: list | None = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (current is None or current[0] != tag[2:])):
            if current is not None:
                spans.append((current[0], current[1], i))
            current = [tag[2:], i]
        elif tag.startswith("I-"):
            continue
        else:
            if current is not None:
                spans.append((current[0], current[1], i))
            current = None
    if current is not None:
        spans.append((current[0], current[1], len(tags)))
    return spans


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def span_prf(pred: Sequence[Sequence[Span]], gold: Sequence[Sequence[Span]]) -> dict[str, object]:
    """Exact-span precision/recall/F1 over sequences of span lists.

    Returns ``{"micro": {...}, "macro": {...}, "per_label": {label: {...}}}``.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for p_spans, g_spans in zip(pred, gold, strict=True):
        p, g = set(p_spans), set(g_spans)
        for label, _, _ in p & g:
            tp[label] += 1
        for label, _, _ in p - g:
            fp[label] += 1
        for label, _, _ in g - p:
            fn[label] += 1
    labels = sorted(set(tp) | set(fp) | set(fn))
    per = {lab: _prf(tp[lab], fp[lab], fn[lab]) for lab in labels}
    micro = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro = {
        "precision": float(np.mean([v["precision"] for v in per.values()])) if per else 0.0,
        "recall": float(np.mean([v["recall"] for v in per.values()])) if per else 0.0,
        "f1": float(np.mean([v["f1"] for v in per.values()])) if per else 0.0,
        "support": micro["support"],
    }
    return {"micro": micro, "macro": macro, "per_label": per}


def exact_match(pred: Sequence[Sequence[int]], gold: Sequence[Sequence[int]]) -> float:
    """Fraction of sequences whose predicted ids equal the gold ids exactly."""
    if not gold:
        return 0.0
    return sum(list(p) == list(g) for p, g in zip(pred, gold, strict=True)) / len(gold)


def token_accuracy(pred: np.ndarray, gold: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Accuracy over tokens where ``mask`` is true (or over all tokens)."""
    p, g = np.asarray(pred), np.asarray(gold)
    m = np.ones_like(g, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    total = int(m.sum())
    return float(((p == g) & m).sum() / total) if total else 0.0


def confusion(pred: np.ndarray, gold: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """``[k, k]`` counts indexed ``[gold, pred]``."""
    p, g = np.asarray(pred).ravel(), np.asarray(gold).ravel()
    m = np.ones_like(g, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    out = np.zeros((k, k), dtype=np.int64)
    np.add.at(out, (g[m], p[m]), 1)
    return out


def format_prf_table(result: dict[str, object]) -> str:
    """Markdown table for MODEL_CARD.md."""
    per = result["per_label"]
    assert isinstance(per, dict)
    lines = ["| Label | P | R | F1 | Support |", "|---|---|---|---|---|"]
    for lab, v in per.items():
        lines.append(f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    micro = result["micro"]
    assert isinstance(micro, dict)
    lines.append(f"| **micro** | {micro['precision']:.3f} | {micro['recall']:.3f} | **{micro['f1']:.3f}** | {micro['support']} |")
    return "\n".join(lines)
