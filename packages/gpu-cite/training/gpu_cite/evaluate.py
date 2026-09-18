"""Evaluate a checkpoint on the synthetic held-out set, the real anystyle and GROBID corpora,
and the hand-written unfamiliar set. ``uv run python -m gpu_cite.evaluate [--run default]``.

Writes ``runs/<run>/eval.json`` and prints the markdown tables used in MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import torch
from gpu_utils_training.features import tokenize

from .data import CACHE, read_jsonl
from .evalsets import load_anystyle, load_grobid, load_unfamiliar
from .export import RUNS, load_model
from .features import featurize_tokens
from .labels import ROLES, TYPES, spans_from_tags
from .metrics import SpanScorer, _utf16, format_table, normalize_span
from .model import CiteTagger, constrained_transitions
from .train import evaluate_rows, viterbi_batch

ANYSTYLE_ROLES = ["AUTHOR", "TITLE", "CONTAINER", "YEAR", "VOLUME", "PAGES", "PUBLISHER", "LOCATION", "EDITOR", "URL", "DOI", "EDITION"]
GROBID_ROLES = ["AUTHOR", "TITLE", "CONTAINER", "YEAR", "VOLUME", "ISSUE", "PAGES", "PUBLISHER", "LOCATION", "EDITOR", "DOI", "ARXIV", "URL"]


@torch.no_grad()
def predict(model: CiteTagger, texts: list[str], batch_size: int = 128) -> list[dict[str, Any]]:
    trans = constrained_transitions(model.transitions())
    out: list[dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        toks = [tokenize(t) for t in chunk]
        rows = [featurize_tokens(t, tk) for t, tk in zip(chunk, toks, strict=True)]
        T = max(1, max(len(r) for r in rows))
        r = torch.zeros(len(chunk), T, len(rows[0][0]) if rows[0] else 12, dtype=torch.long)
        m = torch.zeros(len(chunk), T)
        for j, rr in enumerate(rows):
            if rr:
                r[j, : len(rr)] = torch.as_tensor(rr)
                m[j, : len(rr)] = 1.0
        tags, parts, ty = model(r, m)
        paths = viterbi_batch(tags, trans, m)
        pt = parts.argmax(-1)
        for j in range(len(chunk)):
            n = len(rows[j])
            out.append({"tags": paths[j][:n] if n else [], "parts": [int(pt[j, k]) for k in range(n)], "type": int(ty[j].argmax()), "tokens": toks[j]})
    return out


def score_cases(model: CiteTagger, cases: list[dict[str, Any]], roles: list[str], with_type: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    ap.add_argument("--show-misses", type=int, default=15)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, _ = load_model(args.run)
    model.quant = True
    results: dict[str, Any] = {}

    held = read_jsonl(CACHE / "heldout.jsonl.gz")
    from .train import featurize_set

    held = featurize_set(held, "feats_heldout")
    results["heldout"] = evaluate_rows(model, held)
    print("## Held-out synthetic\n", format_table(results["heldout"]), json.dumps({k: v for k, v in results["heldout"].items() if k != "fields"}), "\n")
    # per-style / per-type exact match on the held-out set
    by_style: dict[str, list[dict[str, Any]]] = {}
    for ex in held:
        by_style.setdefault(ex["style"], []).append(ex)
    results["heldout_by_style"] = {s: evaluate_rows(model, rows)["exact_match"] for s, rows in sorted(by_style.items())}
    print("exact match by style:", results["heldout_by_style"], "\n")

    unfam = load_unfamiliar()
    results["unfamiliar"], misses = score_cases(model, unfam, ROLES, with_type=True)
    print(f"## Unfamiliar ({len(unfam)} cases)\n", format_table(results["unfamiliar"]), json.dumps({k: v for k, v in results["unfamiliar"].items() if k != "fields"}), "\n")
    results["unfamiliar_misses"] = misses
    for m in misses[: args.show_misses]:
        print("MISS", m["gold_type"], "->", m["pred_type"], "|", m["text"][:110])
        for r, w in m["wrong"].items():
            print("    ", r, "gold", w["gold"], "pred", w["pred"])

    anystyle = load_anystyle()
    results["anystyle"], _ = score_cases(model, anystyle, ANYSTYLE_ROLES, with_type=False)
    print(f"\n## anystyle core ({len(anystyle)} cases)\n", format_table(results["anystyle"]), json.dumps({k: v for k, v in results["anystyle"].items() if k != "fields"}), "\n")

    grobid = load_grobid()
    results["grobid"], _ = score_cases(model, grobid, GROBID_ROLES, with_type=False)
    print(f"## GROBID citation corpus ({len(grobid)} cases)\n", format_table(results["grobid"]), json.dumps({k: v for k, v in results["grobid"].items() if k != "fields"}), "\n")

    results["unfamiliar_types"] = dict(Counter(c["type"] for c in unfam))
    (RUNS / args.run / "eval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
