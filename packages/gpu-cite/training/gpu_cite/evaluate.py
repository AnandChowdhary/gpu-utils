"""Evaluate a checkpoint on the synthetic held-out set, the real anystyle and GROBID corpora,
and the hand-written unfamiliar set. ``uv run python -m gpu_cite.evaluate [--run default]``.

Decoding mirrors ``src/decode.ts``: Viterbi (``gpu_utils_training.decode``) over the
learned CRF transitions plus the hard BIO constraints, argmax over the name-part columns
and over the pooled type logits. Writes ``runs/<run>/eval.json`` and prints the markdown
tables used in MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.features import Token, tokenize
from gpu_utils_training.loop import load_checkpoint
from torch import nn

from .data import CACHE, read_jsonl
from .evalsets import load_anystyle, load_grobid, load_unfamiliar
from .features import WIDTH, featurize_tokens
from .labels import ROLES, TAGS, TYPES, spans_from_tags
from .metrics import (
    SpanScorer,
    _utf16,
    entity_spans,
    field_spans,
    format_table,
    normalize_span,
)
from .model import build, split_tags

RUNS = Path(__file__).resolve().parents[1] / "runs"
ANYSTYLE_ROLES = ["AUTHOR", "TITLE", "CONTAINER", "YEAR", "VOLUME", "PAGES", "PUBLISHER", "LOCATION", "EDITOR", "URL", "DOI", "EDITION"]
GROBID_ROLES = ["AUTHOR", "TITLE", "CONTAINER", "YEAR", "VOLUME", "ISSUE", "PAGES", "PUBLISHER", "LOCATION", "EDITOR", "DOI", "ARXIV", "URL"]

BIO_TRANSITIONS = bio_transitions(TAGS).astype(np.float64)
START = bio_start_mask(TAGS).astype(np.float64)


def constrained_transitions(model: nn.Module) -> np.ndarray:
    """Learned CRF transitions plus the hard BIO constraints, as ``src/decode.ts`` builds them."""
    return model.transitions().detach().numpy().astype(np.float64) + BIO_TRANSITIONS  # type: ignore[operator]


@torch.no_grad()
def predict_rows(model: nn.Module, rows: list[list[list[int]]], batch_size: int = 256) -> list[dict[str, Any]]:
    """Per sequence: ``{"tags": [int], "parts": [int], "type": int}`` from feature rows."""
    trans = constrained_transitions(model)
    padding_id: int = model.padding_id  # type: ignore[assignment]
    out: list[dict[str, Any]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        r, mask = collate(chunk, WIDTH, padding_id)
        res = model(r, mask)
        emissions, part_logits = split_tags(res["tags"])
        em = emissions.numpy().astype(np.float64)
        parts = part_logits.argmax(-1).numpy()
        types = res["pooled"].argmax(-1).tolist()
        for j, seq in enumerate(chunk):
            n = len(seq)
            e = em[j, :n].copy()
            if n:
                e[0] += START
            out.append({"tags": viterbi(e, trans) if n else [], "parts": parts[j, :n].tolist(), "type": types[j]})
    return out


def predict(model: nn.Module, texts: list[str]) -> list[dict[str, Any]]:
    """Featurize raw texts and predict; adds ``"tokens"`` to every result."""
    toks: list[list[Token]] = [tokenize(t) for t in texts]
    rows = [featurize_tokens(t, tk) for t, tk in zip(texts, toks, strict=True)]
    preds = predict_rows(model, rows)
    for p, tk in zip(preds, toks, strict=True):
        p["tokens"] = tk
    return preds


@torch.no_grad()
def evaluate_rows(model: nn.Module, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Held-out metrics on featurized synthetic rows (needs ``rows``, ``tags``, ``parts``, ``type``)."""
    model.eval()
    preds = predict_rows(model, [r["rows"] for r in rows])
    scorer = SpanScorer()
    tok_correct = tok_total = 0
    for ex, p in zip(rows, preds, strict=True):
        toks = tokenize(ex["text"])
        gold, pred = ex["tags"], p["tags"]
        tok_correct += sum(int(a == b) for a, b in zip(gold, pred, strict=True))
        tok_total += len(gold)
        # full-record exact match also requires the given/family split on every name token
        gp = [q for q, t in zip(ex["parts"], gold, strict=True) if t]
        pp = [q for q, t in zip(p["parts"], pred, strict=True) if t]
        scorer.add(
            field_spans(ex["text"], gold, toks),
            field_spans(ex["text"], pred, toks),
            ex["type"],
            p["type"],
            entity_spans(ex["text"], gold, toks, "AUTHOR"),
            entity_spans(ex["text"], pred, toks, "AUTHOR"),
            extra_ok=gold == pred and gp == pp,
        )
    summary = scorer.summary()
    summary["token_accuracy"] = round(tok_correct / max(1, tok_total), 4)
    return summary


def evaluate_flat(model: nn.Module, rows: list[dict[str, Any]]) -> dict[str, float]:
    """The scalar subset of ``evaluate_rows`` for ``gpu_utils_training.loop.train``."""
    return {k: float(v) for k, v in evaluate_rows(model, rows).items() if isinstance(v, (int, float))}


def score_cases(model: nn.Module, cases: list[dict[str, Any]], roles: list[str], with_type: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds = predict(model, [c["text"] for c in cases])
    scorer = SpanScorer()
    misses: list[dict[str, Any]] = []
    for c, p in zip(cases, preds, strict=True):
        t16 = _utf16(c["text"])
        gold = {r: {tuple(s) for s in sp} for r, sp in c["spans"].items()}
        raw = spans_from_tags(p["tags"], p["tokens"])
        pred: dict[str, set[tuple[int, int]]] = {}
        for role in roles:
            ents = [(s, e) for r, s, e in raw if r == role]
            if not ents:
                continue
            if role in ("AUTHOR", "EDITOR"):
                pred[role] = {normalize_span(t16, min(s for s, _ in ents), max(e for _, e in ents))}
            else:
                pred[role] = {normalize_span(t16, s, e) for s, e in ents}
        g_auth = [tuple(x) for x in c.get("author_entities", [])] if "author_entities" in c else None
        p_auth = [normalize_span(t16, s, e) for r, s, e in raw if r == "AUTHOR"] if g_auth is not None else None
        gold_type = TYPES.index(c["type"]) if with_type and c.get("type") in TYPES else None
        scorer.add(gold, pred, gold_type, p["type"] if gold_type is not None else None, g_auth, p_auth, roles=roles)
        wrong = {r: (sorted(gold.get(r, set())), sorted(pred.get(r, set()))) for r in roles if gold.get(r, set()) != pred.get(r, set())}
        if wrong or (gold_type is not None and gold_type != p["type"]):
            misses.append({
                "text": c["text"],
                "gold_type": c.get("type"),
                "pred_type": TYPES[p["type"]],
                "wrong": {r: {"gold": [c["text"][s:e] for s, e in g], "pred": [c["text"][s:e] for s, e in pr]} for r, (g, pr) in wrong.items()},
            })
    return scorer.summary(roles), misses


def main() -> None:
    from .train import featurize_set

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--show-misses", type=int, default=15)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = build()
    load_checkpoint(model, RUNS / args.run / "best.pt")
    model.quant = True  # evaluate exactly what ships
    model.eval()
    results: dict[str, Any] = {}

    held = featurize_set(read_jsonl(CACHE / "heldout.jsonl.gz"), "feats_heldout")
    results["heldout"] = evaluate_rows(model, held)
    print("## Held-out synthetic\n", format_table(results["heldout"]), json.dumps({k: v for k, v in results["heldout"].items() if k not in ("fields", "micro")}), "\n")
    by_style: dict[str, list[dict[str, Any]]] = {}
    for ex in held:
        by_style.setdefault(ex["style"], []).append(ex)
    results["heldout_by_style"] = {s: evaluate_rows(model, rows)["exact_match"] for s, rows in sorted(by_style.items())}
    print("exact match by style:", results["heldout_by_style"], "\n")

    unfam = load_unfamiliar()
    results["unfamiliar"], misses = score_cases(model, unfam, ROLES, with_type=True)
    print(f"## Unfamiliar ({len(unfam)} cases)\n", format_table(results["unfamiliar"]), json.dumps({k: v for k, v in results["unfamiliar"].items() if k not in ("fields", "micro")}), "\n")
    results["unfamiliar_misses"] = misses
    for m in misses[: args.show_misses]:
        print("MISS", m["gold_type"], "->", m["pred_type"], "|", m["text"][:110])
        for r, w in m["wrong"].items():
            print("    ", r, "gold", w["gold"], "pred", w["pred"])

    anystyle = load_anystyle()
    results["anystyle"], _ = score_cases(model, anystyle, ANYSTYLE_ROLES, with_type=False)
    print(f"\n## anystyle core ({len(anystyle)} cases)\n", format_table(results["anystyle"]), json.dumps({k: v for k, v in results["anystyle"].items() if k not in ("fields", "micro")}), "\n")

    grobid = load_grobid()
    results["grobid"], _ = score_cases(model, grobid, GROBID_ROLES, with_type=False)
    print(f"## GROBID citation corpus ({len(grobid)} cases)\n", format_table(results["grobid"]), json.dumps({k: v for k, v in results["grobid"].items() if k not in ("fields", "micro")}), "\n")

    results["unfamiliar_types"] = dict(Counter(c["type"] for c in unfam))
    (RUNS / args.run / "eval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
