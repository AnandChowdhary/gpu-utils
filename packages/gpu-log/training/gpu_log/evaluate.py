"""Evaluation on four sets:

* **held-out** (generated, seed 2): the training distribution;
* **unfamiliar v1** (frozen, hand-written, `../eval/unfamiliar-v1.jsonl`): the v1 hard set.
  It drove the v2 generator widening, so for v2 it is *contaminated* and is reported for
  continuity only;
* **unfamiliar v2** (hand-written, fresh, `data/unfamiliar_v2.txt`): layouts the v2
  generator does not produce, written before any v2 evaluation;
* **real-world** (Loghub, `loghub.py`): real third-party log lines whose gold spans come
  from Loghub's own structured CSVs, not from us. Downloaded at evaluation time.

    uv run python -m gpu_log.evaluate [--run default] [--real-limit 1000] [--baseline]

`--baseline` also scores the promoted v1 model (rebuilt from `main`) on the same sets with
the same code. v1 has no HOST role, so every comparison run folds HOST into O for both
models ("common roles").
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from gpu_utils_training.batch import collate
from gpu_utils_training.decode import bio_start_mask, bio_transitions, viterbi
from gpu_utils_training.loop import load_checkpoint
from gpu_utils_training.metrics import bio_to_spans, span_prf
from gpu_utils_training.qat import set_quant

from .data import (
    DATA_DIR,
    LABELS,
    Dataset,
    Example,
    build,
    cached,
    read_jsonl,
    read_markup_file,
    tag_example,
)
from .features import FEATURE_COUNT, featurize
from .gen import KINDS, ROLES
from .loghub import SEEN_LAYOUTS, SYSTEMS, RealLine
from .loghub import load as loghub_load

EVAL_DIR = DATA_DIR.parents[1] / "eval"
Tagged = tuple[list[str], int]  # BIO label strings per token, kind id


@torch.no_grad()
def predict_lines(model: torch.nn.Module, feats_list: list[list[list[int]]], labels: list[str] = LABELS) -> list[Tagged]:
    """Constrained-Viterbi label strings and the kind id per line; batched by length."""
    model.eval()
    trans = bio_transitions(labels)
    start = bio_start_mask(labels)
    out: list[Tagged | None] = [None] * len(feats_list)
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
            out[i] = ([labels[t] for t in viterbi(em, trans)], int(kinds[j]))
    return out  # type: ignore[return-value]


def fold(tags: list[str], drop_roles: frozenset[str]) -> list[str]:
    """Map labels of dropped roles to O, so label sets of different vintages can be compared."""
    if not drop_roles:
        return tags
    return [t if t == "O" or t[2:] not in drop_roles else "O" for t in tags]


def score(
    pred: list[Tagged],
    gold_tags: list[list[str]],
    gold_kinds: list[int],
    masks: list[list[bool]] | None = None,
    dropped: list[set[tuple[int, int]]] | None = None,
    drop_roles: frozenset[str] = frozenset(),
) -> dict:
    """Token accuracy, span P/R/F1, kind accuracy and line-exact.

    `masks[i][t]` False excludes a token from token accuracy and line-exact; `dropped[i]`
    holds token ranges the gold does not label, so predictions falling entirely inside one
    are discarded instead of counted as false positives.
    """
    tok_correct = tok_total = 0
    kind_correct = line_exact = 0
    pred_spans, gold_spans = [], []
    kind_conf: Counter[tuple[str, str]] = Counter()
    for i, ((ptags, pkind), gold, gkind) in enumerate(zip(pred, gold_tags, gold_kinds, strict=True)):
        ptags, gtags = fold(ptags, drop_roles), fold(gold, drop_roles)
        eq = np.asarray(ptags) == np.asarray(gtags)
        keep = np.asarray(masks[i], dtype=bool) if masks is not None else np.ones(len(gtags), dtype=bool)
        tok_correct += int((eq & keep).sum())
        tok_total += int(keep.sum())
        kind_correct += int(pkind == gkind)
        kind_conf[(KINDS[gkind], KINDS[pkind])] += 1
        if bool((eq | ~keep).all()) and pkind == gkind:
            line_exact += 1
        spans = bio_to_spans(ptags)
        if dropped is not None:
            spans = [sp for sp in spans if not any(lo <= sp[1] and sp[2] <= hi for lo, hi in dropped[i])]
        pred_spans.append(spans)
        gold_spans.append(bio_to_spans(gtags))
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


def evaluate_dataset(model: torch.nn.Module, ds: Dataset, limit: int | None = None, labels: list[str] = LABELS,
                     drop_roles: frozenset[str] = frozenset()) -> dict:
    n = len(ds) if limit is None else min(limit, len(ds))
    feats = [ds.line(i)[0].tolist() for i in range(n)]
    gold_tags = [[LABELS[t] for t in ds.line(i)[1].astype(np.int64).tolist()] for i in range(n)]
    gold_kinds = [ds.line(i)[2] for i in range(n)]
    return score(predict_lines(model, feats, labels), gold_tags, gold_kinds, drop_roles=drop_roles)


def evaluate_examples(model: torch.nn.Module, examples: list[Example], labels: list[str] = LABELS,
                      drop_roles: frozenset[str] = frozenset()) -> dict:
    ds, _ = build(examples)
    return evaluate_dataset(model, ds, labels=labels, drop_roles=drop_roles)


def print_report(name: str, m: dict) -> None:
    f = m["flat"]
    print(f"\n== {name}: {m['lines']} lines")
    print(f"token acc {f['token_acc']:.4f} | span F1 {f['span_f1']:.4f} | kind acc {f['kind_acc']:.4f} | line exact {f['line_exact']:.4f}")
    for r in ROLES:
        v = m["per_role"].get(r)
        if v and v["support"]:
            print(f"  {r:7s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (n={v['support']})")


# --- Real-world set: Loghub lines with gold spans from Loghub's structured CSVs ----------


def real_lines(limit: int = 1000) -> list[RealLine]:
    lines, coverage = loghub_load(limit)
    missing = [s for s in SYSTEMS if s not in coverage]
    if missing:
        print(f"  (no Loghub data for {', '.join(missing)}; offline?)")
    return lines


def prepare_real(lines: list[RealLine]) -> tuple[list, list, list, list]:
    """(feature rows, gold labels, scored-token masks, unlabelled token ranges) per line."""
    feats, golds, masks, dropped = [], [], [], []
    for rl in lines:
        rows, tags, _kind, _mis = tag_example(Example(rl.text, rl.spans, "entry"))
        tokens, _ = featurize(rl.text)
        u16 = [0]
        for ch in rl.text:
            u16.append(u16[-1] + (2 if ord(ch) > 0xFFFF else 1))
        ignores = [(u16[s], u16[e]) for s, e in rl.ignores]
        keep = [not any(lo < t.end and t.start < hi for lo, hi in ignores) for t in tokens]
        ranges: list[tuple[int, int]] = []
        run: int | None = None
        for i, k in enumerate(keep):
            if not k and run is None:
                run = i
            elif k and run is not None:
                ranges.append((run, i))
                run = None
        if run is not None:
            ranges.append((run, len(keep)))
        feats.append(rows)
        golds.append([LABELS[t] for t in tags])
        masks.append(keep)
        dropped.append(set(ranges))
    return feats, golds, masks, dropped


def evaluate_real(model: torch.nn.Module, lines: list[RealLine], labels: list[str] = LABELS,
                  drop_roles: frozenset[str] = frozenset()) -> tuple[dict, dict[str, dict]]:
    feats, golds, masks, dropped = prepare_real(lines)
    pred = predict_lines(model, feats, labels)
    kinds = [KINDS.index("entry")] * len(lines)

    def sub(idx: list[int]) -> dict:
        return score([pred[i] for i in idx], [golds[i] for i in idx], [kinds[i] for i in idx],
                     [masks[i] for i in idx], [dropped[i] for i in idx], drop_roles)

    per_system = {s: sub([i for i, rl in enumerate(lines) if rl.system == s])
                  for s in dict.fromkeys(rl.system for rl in lines)}
    for name, want in (("* layouts the generator imitates", True), ("* unseen layouts", False)):
        idx = [i for i, rl in enumerate(lines) if (rl.system in SEEN_LAYOUTS) == want]
        if idx:
            per_system[name] = sub(idx)
    return sub(list(range(len(lines)))), per_system


def load_model(run: str) -> torch.nn.Module:
    from .model import build as build_model

    model = build_model()
    load_checkpoint(model, DATA_DIR.parent / "runs" / run / "best.pt")
    set_quant(model, True)
    model.eval()
    return model


def report_all(name: str, model: torch.nn.Module, labels: list[str], real: list[RealLine],
               drop_roles: frozenset[str] = frozenset()) -> None:
    tag = f" [{name}]" if name else ""
    kw = {"labels": labels, "drop_roles": drop_roles}
    print_report(f"held-out (generated, seed 2){tag}", evaluate_dataset(model, cached("heldout", 12_000, 2), **kw))
    v1_path = EVAL_DIR / "unfamiliar-v1.jsonl"
    if v1_path.exists():
        m = evaluate_examples(model, read_jsonl(v1_path), **kw)
        print_report(f"unfamiliar v1 (frozen; contaminated for v2: it drove the v2 coverage){tag}", m)
        print("  kind confusion:", m["kind_confusion"])
    v2_path = DATA_DIR / "unfamiliar_v2.txt"
    if v2_path.exists():
        m = evaluate_examples(model, read_markup_file(v2_path), **kw)
        print_report(f"unfamiliar v2 (hand-written, fresh){tag}", m)
        print("  kind confusion:", m["kind_confusion"])
    if real:
        overall, per_system = evaluate_real(model, real, **kw)
        print_report(f"real-world (Loghub 2k samples, gold spans from the structured CSVs){tag}", overall)
        print(f"\n== real-world per system{tag}")
        for system, m in sorted(per_system.items()):
            f = m["flat"]
            seen = "" if system.startswith("*") else ("  (seen layout)" if system in SEEN_LAYOUTS else "  (UNSEEN layout)")
            print(f"  {system:32s} {m['lines']:6d} lines  token acc {f['token_acc']:.4f}  span F1 {f['span_f1']:.4f}  line exact {f['line_exact']:.4f}{seen}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate gpu-log")
    ap.add_argument("--run", default="default")
    ap.add_argument("--real-limit", type=int, default=1000, help="Loghub lines per system")
    ap.add_argument("--baseline", action="store_true", help="also score the v1 model from main")
    args = ap.parse_args()
    torch.set_num_threads(2)
    real = real_lines(args.real_limit)
    model = load_model(args.run)
    report_all("v2, all roles" if args.baseline else "", model, LABELS, real)
    if args.baseline:
        from .baseline_v1 import load_v1

        v1, v1_labels = load_v1()
        common = frozenset({"HOST"})
        print("\n\n######## common-role scoring (HOST folded into O; v1 has no HOST role) ########")
        report_all("v2, common roles", model, LABELS, real, common)
        report_all("v1, common roles", v1, v1_labels, real, common)


if __name__ == "__main__":
    main()
