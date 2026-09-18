"""Evaluation: held-out generated set, frozen v1 unfamiliar set (contaminated: it drove the v2
coverage), fresh v2 unfamiliar set, and Loghub real samples (weak labels).

    uv run python -m gpu_log.evaluate [--run default]
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.loop import load_checkpoint
from gpu_utils_training.metrics import bio_to_spans, span_prf
from gpu_utils_training.qat import set_quant

from .data import DATA_DIR, LABELS, Dataset, build, cached, read_jsonl, read_markup_file
from .features import FEATURE_COUNT, featurize
from .gen import KINDS, ROLES

EVAL_DIR = DATA_DIR.parents[1] / "eval"


@torch.no_grad()
def predict_lines(model: torch.nn.Module, feats_list: list[list[list[int]]]) -> list[tuple[list[int], int]]:
    """(Viterbi tag ids, kind id) per line; batched by length."""
    model.eval()
    trans = bio_transitions(LABELS)
    start = bio_start_mask(LABELS)
    out: list[tuple[list[int], int] | None] = [None] * len(feats_list)
    order = sorted(range(len(feats_list)), key=lambda i: len(feats_list[i]))
    for b in range(0, len(order), 64):
        idx = order[b : b + 64]
        rows, mask = collate([feats_list[i] or [[model.padding_id] * FEATURE_COUNT] for i in idx], FEATURE_COUNT, model.padding_id)
        res = model(rows, mask)
        tags = res["tags"].numpy()
        kinds = res["pooled"].numpy().argmax(1)
        for j, i in enumerate(idx):
            n = len(feats_list[i])
            if n == 0:
                out[i] = ([], int(kinds[j]))
                continue
            em = tags[j, :n].astype(np.float64)
            em[0] += start
            out[i] = (viterbi(em, trans), int(kinds[j]))
    return out  # type: ignore[return-value]


def score(pred: list[tuple[list[int], int]], gold_tags: list[list[int]], gold_kinds: list[int]) -> dict:
    tok_correct = tok_total = 0
    kind_correct = line_exact = 0
    pred_spans, gold_spans = [], []
    kind_conf: Counter[tuple[str, str]] = Counter()
    for (ptags, pkind), gtags, gkind in zip(pred, gold_tags, gold_kinds, strict=True):
        eq = np.asarray(ptags) == np.asarray(gtags)
        tok_correct += int(eq.sum())
        tok_total += len(gtags)
        kind_correct += int(pkind == gkind)
        kind_conf[(KINDS[gkind], KINDS[pkind])] += 1
        if eq.all() and pkind == gkind:
            line_exact += 1
        pred_spans.append(bio_to_spans([LABELS[t] for t in ptags]))
        gold_spans.append(bio_to_spans([LABELS[t] for t in gtags]))
    prf = span_prf(pred_spans, gold_spans)
    n = len(gold_kinds)
    flat = {
        "token_acc": tok_correct / max(1, tok_total),
        "span_f1": prf["micro"]["f1"],
        "kind_acc": kind_correct / max(1, n),
        "line_exact": line_exact / max(1, n),
    }
    return {"lines": n, "flat": flat, "per_role": prf["per_label"], "kind_confusion": dict(kind_conf),
            "summary": " ".join(f"{k} {v:.4f}" for k, v in flat.items())}


def evaluate_dataset(model: torch.nn.Module, ds: Dataset, limit: int | None = None) -> dict:
    n = len(ds) if limit is None else min(limit, len(ds))
    feats = [ds.line(i)[0].tolist() for i in range(n)]
    gold_tags = [ds.line(i)[1].astype(np.int64).tolist() for i in range(n)]
    gold_kinds = [ds.line(i)[2] for i in range(n)]
    return score(predict_lines(model, feats), gold_tags, gold_kinds)


def evaluate_examples(model: torch.nn.Module, examples) -> dict:
    ds, _ = build(examples)
    return evaluate_dataset(model, ds)


def print_report(name: str, m: dict) -> None:
    f = m["flat"]
    print(f"\n== {name}: {m['lines']} lines")
    print(f"token acc {f['token_acc']:.4f} | span F1 {f['span_f1']:.4f} | kind acc {f['kind_acc']:.4f} | line exact {f['line_exact']:.4f}")
    for r in ROLES:
        v = m["per_role"].get(r)
        if v and v["support"]:
            print(f"  {r:7s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")


# --- Loghub (research-license, evaluation only, downloaded at runtime) ------------------

LOGHUB_RAW = "https://raw.githubusercontent.com/logpai/loghub/master/{d}/{d}_2k.log{suffix}"
LOGHUB_SETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "OpenSSH", "Linux", "Mac", "Apache", "Android", "BGL", "HPC",
               "Thunderbird", "Windows", "HealthApp", "Proxifier", "OpenStack"]
LOGHUB_FIELDS = {"Level": "level", "Component": "source", "Content": "message"}
LEVEL_NORMAL = {"info": "info", "warn": "warn", "warning": "warn", "error": "error", "err": "error", "fatal": "fatal",
                "debug": "debug", "trace": "trace", "notice": "info", "severe": "error", "critical": "error", "crit": "error",
                "v": "trace", "d": "debug", "i": "info", "w": "warn", "e": "error", "f": "fatal"}


def fetch(d: str, suffix: str, cache: Path) -> Path | None:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{d}_2k.log{suffix}"
    if not p.exists():
        try:
            with urllib.request.urlopen(LOGHUB_RAW.format(d=d, suffix=suffix), timeout=30) as r:
                p.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  (skipping {d}: {e})")
            return None
    return p


def evaluate_loghub(model: torch.nn.Module, cache: Path, limit: int = 300) -> dict[str, dict]:
    """Weak-label agreement on real logs against the structured CSV columns. Loghub's
    "Component" is sometimes the hostname (OpenSSH) and its "Content" sometimes keeps a
    leading id, so this is a sanity signal rather than a score."""
    results = {}
    for d in LOGHUB_SETS:
        raw_path, csv_path = fetch(d, "", cache), fetch(d, "_structured.csv", cache)
        if raw_path is None or csv_path is None:
            continue
        lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()[:limit]
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8", errors="replace"))))[:limit]
        toks = [featurize(ln) for ln in lines]
        preds = predict_lines(model, [t[1] for t in toks])
        hits: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        for ln, (tokens, _), row, (tags, _kind) in zip(lines, toks, rows, preds, strict=True):
            got: dict[str, list[str]] = defaultdict(list)
            for role, s, e in sorted(bio_to_spans([LABELS[t] for t in tags]), key=lambda x: x[1]):
                got[role].append(ln[tokens[s].start : tokens[e - 1].end])
            for col, field in LOGHUB_FIELDS.items():
                gold = (row.get(col) or "").strip()
                if not gold:
                    continue
                totals[field] += 1
                if field == "level":
                    g = LEVEL_NORMAL.get(gold.lower(), gold.lower())
                    p = LEVEL_NORMAL.get(got["LEVEL"][0].lower(), got["LEVEL"][0].lower()) if got["LEVEL"] else ""
                    hits[field] += int(g == p)
                elif field == "source":
                    cands = got["SOURCE"] + got["HOST"]
                    hits[field] += int(any(gold == s or gold in s or s in gold for s in cands) if cands else 0)
                else:
                    msg = " ".join(got["MSG"]).strip()
                    hits[field] += int(bool(msg) and (gold == msg or gold.startswith(msg) or msg.startswith(gold[:40])))
        results[d] = {f: (hits[f], totals[f]) for f in ("level", "source", "message") if totals[f]}
    return results


def load_model(run: str) -> torch.nn.Module:
    from .model import build as build_model

    model = build_model()
    load_checkpoint(model, DATA_DIR.parent / "runs" / run / "best.pt")
    set_quant(model, True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="default")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.run)
    print_report("held-out (generated, seed 2)", evaluate_dataset(model, cached("heldout", 12_000, 2)))
    v1 = EVAL_DIR / "unfamiliar-v1.jsonl"
    if v1.exists():
        m = evaluate_examples(model, read_jsonl(v1))
        print_report("unfamiliar v1 (frozen, contaminated: drove v2 coverage)", m)
        print("  kind confusion:", m["kind_confusion"])
    v2 = DATA_DIR / "unfamiliar_v2.txt"
    if v2.exists():
        m = evaluate_examples(model, read_markup_file(v2))
        print_report("unfamiliar v2 (hand-written, fresh)", m)
        print("  kind confusion:", m["kind_confusion"])
    print("\n== Loghub 2k samples (weak labels from *_structured.csv; first 300 lines each)")
    for d, r in evaluate_loghub(model, DATA_DIR / "cache" / "loghub").items():
        print(f"  {d:12s} " + "  ".join(f"{f} {h}/{t} ({h / t:.0%})" for f, (h, t) in r.items()))


if __name__ == "__main__":
    main()
