"""Evaluation: held-out generated set, hand-written unfamiliar set, Loghub real samples.

`uv run python -m gpu_log.evaluate [runs/latest.pt]`
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import DATA_DIR, KIND_INDEX, LABELS, Dataset, build, cached, read_markup_file
from .features import featurize
from .gen import KINDS, ROLES
from .model import LogTagger

TRANSITIONS = None


def transitions() -> np.ndarray:
    """BIO constraints: I-X may only follow B-X or I-X."""
    global TRANSITIONS
    if TRANSITIONS is None:
        k = len(LABELS)
        t = np.zeros((k, k), dtype=np.float32)
        for j, to in enumerate(LABELS):
            if to.startswith("I-"):
                role = to[2:]
                for i, frm in enumerate(LABELS):
                    if frm not in (f"B-{role}", f"I-{role}"):
                        t[i, j] = -1e9
        TRANSITIONS = t
    return TRANSITIONS


def viterbi(em: np.ndarray) -> np.ndarray:
    n, k = em.shape
    t = transitions()
    score = em[0].astype(np.float64).copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        cand = score[:, None] + t
        back[i] = cand.argmax(0)
        score = cand.max(0) + em[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def spans_from_tags(tags: list[int]) -> set[tuple[int, int, str]]:
    out = set()
    start, role = -1, ""
    for i, t in enumerate(list(tags) + [0]):
        lab = LABELS[t] if t < len(LABELS) else "O"
        if lab == "O" or lab.startswith("B-") or (lab.startswith("I-") and lab[2:] != role):
            if role:
                out.add((start, i, role))
            start, role = (i, lab[2:]) if lab != "O" else (-1, "")
    return out


@torch.no_grad()
def predict_lines(model: LogTagger, feats_list: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    """Returns (tag path, kind) per line, batching by length."""
    model.eval()
    out: list[tuple[np.ndarray, int] | None] = [None] * len(feats_list)
    order = sorted(range(len(feats_list)), key=lambda i: len(feats_list[i]))
    for b in range(0, len(order), 64):
        idx = order[b : b + 64]
        L = max(1, max(len(feats_list[i]) for i in idx))
        F_ = feats_list[idx[0]].shape[1] if len(feats_list[idx[0]]) else 9
        feats = np.zeros((len(idx), L, F_), dtype=np.int64)
        mask = np.zeros((len(idx), L), dtype=np.float32)
        for j, i in enumerate(idx):
            n = len(feats_list[i])
            feats[j, :n] = feats_list[i]
            mask[j, :n] = 1.0
        tags, kinds = model(torch.from_numpy(feats), torch.from_numpy(mask))
        tags = tags.numpy()
        kinds = kinds.numpy().argmax(1)
        for j, i in enumerate(idx):
            n = len(feats_list[i])
            out[i] = (viterbi(tags[j, :n]) if n else np.zeros(0, dtype=np.int64), int(kinds[j]))
    return out  # type: ignore[return-value]


def score(pred: list[tuple[np.ndarray, int]], gold_tags: list[np.ndarray], gold_kinds: list[int]) -> dict:
    tok_correct = tok_total = 0
    kind_correct = 0
    line_exact = 0
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    kind_conf: Counter[tuple[str, str]] = Counter()
    for (ptags, pkind), gtags, gkind in zip(pred, gold_tags, gold_kinds, strict=True):
        eq = ptags == gtags
        tok_correct += int(eq.sum())
        tok_total += len(gtags)
        kind_correct += int(pkind == gkind)
        kind_conf[(KINDS[gkind], KINDS[pkind])] += 1
        if eq.all() and pkind == gkind:
            line_exact += 1
        ps, gs = spans_from_tags(list(ptags)), spans_from_tags(list(gtags))
        for s in ps & gs:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1
    per_role = {}
    for r in ROLES:
        p = tp[r] / max(1, tp[r] + fp[r])
        rc = tp[r] / max(1, tp[r] + fn[r])
        per_role[r] = {"p": p, "r": rc, "f1": 2 * p * rc / max(1e-9, p + rc), "support": tp[r] + fn[r]}
    micro_p = sum(tp.values()) / max(1, sum(tp.values()) + sum(fp.values()))
    micro_r = sum(tp.values()) / max(1, sum(tp.values()) + sum(fn.values()))
    n = len(gold_kinds)
    return {
        "lines": n,
        "token_acc": tok_correct / max(1, tok_total),
        "kind_acc": kind_correct / max(1, n),
        "line_exact": line_exact / max(1, n),
        "span_f1": 2 * micro_p * micro_r / max(1e-9, micro_p + micro_r),
        "per_role": per_role,
        "kind_confusion": dict(kind_conf),
        "summary": f"tok {tok_correct / max(1, tok_total):.4f} span-F1 {2 * micro_p * micro_r / max(1e-9, micro_p + micro_r):.4f} kind {kind_correct / max(1, n):.4f} line {line_exact / max(1, n):.4f}",
    }


def evaluate_dataset(model: LogTagger, ds: Dataset, limit: int | None = None) -> dict:
    n = len(ds) if limit is None else min(limit, len(ds))
    feats = [ds.line(i)[0] for i in range(n)]
    gold_tags = [ds.line(i)[1].astype(np.int64) for i in range(n)]
    gold_kinds = [ds.line(i)[2] for i in range(n)]
    return score(predict_lines(model, feats), gold_tags, gold_kinds)


def print_report(name: str, m: dict) -> None:
    print(f"\n== {name}: {m['lines']} lines")
    print(f"token acc {m['token_acc']:.4f} | span F1 {m['span_f1']:.4f} | kind acc {m['kind_acc']:.4f} | line exact {m['line_exact']:.4f}")
    for r, v in m["per_role"].items():
        if v["support"]:
            print(f"  {r:7s} P {v['p']:.3f} R {v['r']:.3f} F1 {v['f1']:.3f} (n={v['support']})")


# --- Loghub (research-license, evaluation only, downloaded at runtime) ------------------

LOGHUB = "https://raw.githubusercontent.com/logpai/loghub/master/{d}/{d}_2k.log_structured.csv"
LOGHUB_SETS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "OpenSSH", "Linux", "Mac", "Apache", "Android", "BGL", "HPC",
               "Thunderbird", "Windows", "HealthApp", "Proxifier", "OpenStack"]
# Which CSV columns correspond to which of our fields (Content ~ message).
LOGHUB_FIELDS = {"Level": "level", "Component": "source", "Content": "message"}
LEVEL_NORMAL = {"info": "info", "warn": "warn", "warning": "warn", "error": "error", "err": "error", "fatal": "fatal",
                "debug": "debug", "trace": "trace", "notice": "info", "severe": "error", "critical": "error", "crit": "error",
                "v": "trace", "d": "debug", "i": "info", "w": "warn", "e": "error", "f": "fatal"}


def fetch_loghub(d: str, cache: Path) -> list[dict] | None:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{d}_2k.log_structured.csv"
    if not p.exists():
        try:
            with urllib.request.urlopen(LOGHUB.format(d=d), timeout=30) as r:
                p.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  (skipping {d}: {e})")
            return None
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8", errors="replace"))))


def line_from_row(d: str, row: dict) -> str | None:
    """Reconstruct the raw line for datasets whose CSV keeps the whole content (best effort)."""
    raw = row.get("Line") or row.get("Raw")
    if raw:
        return raw
    return None


def evaluate_loghub(model: LogTagger, cache: Path, limit: int = 300) -> dict[str, dict]:
    """Weak-label check on real logs: compare the model's level/source/message against the
    structured CSV columns. The raw line is read from <D>_2k.log (same order as the CSV)."""
    results = {}
    for d in LOGHUB_SETS:
        rows = fetch_loghub(d, cache)
        raw_path = cache / f"{d}_2k.log"
        if rows is None:
            continue
        if not raw_path.exists():
            try:
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/logpai/loghub/master/{d}/{d}_2k.log", timeout=30) as r:
                    raw_path.write_bytes(r.read())
            except Exception as e:  # noqa: BLE001
                print(f"  (skipping {d}: {e})")
                continue
        lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = rows[:limit]
        lines = lines[:limit]
        feats = [featurize(ln)[1] for ln in lines]
        preds = predict_lines(model, [np.asarray(f, dtype=np.int64).reshape(-1, 9) for f in feats])
        hits: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        for ln, row, (tags, _kind) in zip(lines, rows, preds, strict=True):
            tokens = featurize(ln)[0]
            got: dict[str, list[str]] = defaultdict(list)
            for s, e, role in sorted(spans_from_tags(list(tags))):
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
                    hits[field] += int(any(gold == s or gold in s or s in gold for s in got["SOURCE"]) if got["SOURCE"] else 0)
                else:
                    msg = " ".join(got["MSG"]).strip()
                    hits[field] += int(bool(msg) and (gold.strip() == msg or gold.strip().startswith(msg) or msg.startswith(gold.strip()[:40])))
        results[d] = {f: (hits[f], totals[f]) for f in ("level", "source", "message") if totals[f]}
    return results


def load_model(path: Path) -> LogTagger:
    ck = torch.load(path, map_location="cpu")
    model = LogTagger()
    model.load_state_dict(ck["state"])
    model.set_quant(True)
    model.eval()
    return model


def evaluate_markup(model: LogTagger, path: Path) -> dict:
    exs = read_markup_file(path)
    ds, _ = build(exs)
    return evaluate_dataset(model, ds)


def main() -> None:
    torch.set_num_threads(2)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "runs" / "latest.pt"
    model = load_model(path)
    print_report("held-out (generated, seed 2)", evaluate_dataset(model, cached("heldout", 12_000, 2)))
    unf = DATA_DIR / "unfamiliar.txt"
    if unf.exists():
        m = evaluate_markup(model, unf)
        print_report("unfamiliar (hand-written)", m)
        print("  kind confusion:", m["kind_confusion"])
    print("\n== Loghub 2k samples (weak labels from *_structured.csv; first 300 lines each)")
    for d, r in evaluate_loghub(model, DATA_DIR / "cache" / "loghub").items():
        print(f"  {d:12s} " + "  ".join(f"{f} {h}/{t} ({h / t:.0%})" for f, (h, t) in r.items()))


if __name__ == "__main__":
    main()
