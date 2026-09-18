"""Corpora, batches and the loss; also writes the eval sets and cross-language fixtures.

  uv run python -m gpu_view.data            # fixtures, ../eval/*.json, data/cache/*.npz

Training data is generated (never downloaded): there is no permissively licensed corpus
of search-bar phrases paired with table-view specs, and the schema-blind setup needs the
schema that produced each phrase. The transfer set is generated from four domains whose
vocabulary is disjoint from training (schema.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import torch
from gpu_utils_training.batch import collate, pad_labels
from torch import Tensor, nn
from torch.nn import functional as F

from . import features, generate, match, schema as schema_module, timeres
from .generate import ROLES, dataset, example_json

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data" / "cache"
PACKAGE = HERE.parent.parent
EVAL_DIR = PACKAGE / "eval"
FIXTURES = PACKAGE / "test" / "fixtures"
MAX_LEN = 48
PADDING_ID = features.PADDING_ROW


def encode(examples: list[generate.Example]) -> dict[str, np.ndarray]:
    """Fixed-shape arrays: rows [N, MAX_LEN, SLOTS], roles, bounds, valid."""
    n = len(examples)
    rows = np.full((n, MAX_LEN, features.SLOTS), PADDING_ID, dtype=np.int32)
    roles = np.zeros((n, MAX_LEN), dtype=np.int64)
    bounds = np.zeros((n, MAX_LEN), dtype=np.float32)
    valid = np.zeros((n, MAX_LEN), dtype=bool)
    for k, ex in enumerate(examples):
        toks, r = features.featurize(ex.text, ex.schema)
        assert [t.text for t in toks] == ex.tokens
        t = min(len(r), MAX_LEN)
        for i in range(t):
            rows[k, i, : len(r[i])] = r[i]
        roles[k, :t] = ex.roles[:t]
        bounds[k, :t] = ex.boundaries[:t]
        valid[k, :t] = True
    return {"rows": rows, "roles": roles, "bounds": bounds, "valid": valid}


def build(samples: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    schema_module.assert_disjoint()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    specs = {"train": ("train", samples, seed), "dev": ("train", 2000, seed + 1000), "transfer": ("eval", 2000, seed + 2000)}
    for name, (split, count, s) in specs.items():
        path = CACHE / f"{name}-{count}-{s}.npz"
        if path.exists():
            out[name] = dict(np.load(path))
            continue
        enc = encode(dataset(split, count, s))
        np.savez_compressed(path, **enc)
        out[name] = enc
        print(f"{name}: {count} examples → {path.name}")
    return out


Batch = tuple[Tensor, Tensor, Tensor, Tensor]


def batches(enc: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator | None) -> list[Batch]:
    """Shuffled batches of (rows, mask, role ids, boundary targets), trimmed to the longest row."""
    n = enc["rows"].shape[0]
    order = rng.permutation(n) if rng is not None else np.arange(n)
    out: list[Batch] = []
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        valid = enc["valid"][idx]
        t = int(valid.any(axis=0).nonzero()[0].max()) + 1 if valid.any() else 1
        out.append((
            torch.from_numpy(enc["rows"][idx, :t].astype(np.int64)),
            torch.from_numpy(valid[:, :t]),
            torch.from_numpy(enc["roles"][idx, :t]),
            torch.from_numpy(enc["bounds"][idx, :t]),
        ))
    return out


def loss(model: nn.Module, batch: Batch) -> Tensor:
    rows, mask, roles, bounds = batch
    logits = model(rows, mask)["tags"]
    k = len(ROLES)
    v = mask.reshape(-1)
    ce = F.cross_entropy(logits[..., :k].reshape(-1, k)[v], roles.reshape(-1)[v])
    bce = F.binary_cross_entropy_with_logits(logits[..., k].reshape(-1)[v], bounds.reshape(-1)[v], pos_weight=torch.tensor(2.0))
    return ce + 0.5 * bce


def write_eval_sets(seed: int = 4242) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    transfer = [example_json(e) for e in dataset("eval", 400, seed)]
    dev = [example_json(e) for e in dataset("train", 200, seed + 1)]
    (EVAL_DIR / "heldout.json").write_text(json.dumps(transfer, ensure_ascii=False) + "\n")
    (EVAL_DIR / "indomain.json").write_text(json.dumps(dev, ensure_ascii=False) + "\n")
    print(f"eval sets: {len(transfer)} transfer, {len(dev)} in-domain → {EVAL_DIR}")


def write_fixtures() -> None:
    """Cross-language parity fixtures for the featurizer, matcher and time resolver."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    cases = []
    for ex in dataset("train", 40, 77) + dataset("eval", 20, 78):
        toks, rows = features.featurize(ex.text, ex.schema)
        cases.append({"text": ex.text, "schema": ex.schema, "tokens": [t.text for t in toks], "rows": rows})
    extra_schema = {"fields": [
        {"name": "customer", "kind": "text", "aliases": ["buyer"]},
        {"name": "country", "kind": "enum", "values": ["Germany", "France", "New Zealand"]},
        {"name": "orders", "kind": "number", "aliases": ["order count"]},
        {"name": "createdAt", "kind": "date", "aliases": ["signed up", "created"]},
        {"name": "status", "kind": "enum", "values": ["in-progress", "done"]},
        {"name": "vip", "kind": "boolean"},
        {"name": "price", "kind": "number", "aliases": ["cheap", "rated"]},
    ]}
    for text in ["customers in Germany or France with more than 5 orders", "in-progress vip buyers",
                 "New Zealand custmer signed-up last 30 days", "orderz > 5k", "CREATED AT 2024-01-01",
                 "café 😀 x", "Status: In Progress, country=germany", "", "   ", "createdAt",
                 "cheapest first", "highest rated, taller than 50 cm", "top 5 buyers by price"]:
        toks, rows = features.featurize(text, extra_schema)
        cases.append({"text": text, "schema": extra_schema, "tokens": [t.text for t in toks], "rows": rows})
    (FIXTURES / "features.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")

    today = date(2026, 9, 17)
    phrases = ["today", "yesterday", "tomorrow", "now", "ytd", "last 30 days", "past 7 days",
               "previous 2 weeks", "last 3 months", "last 2 years", "next 7 days", "next 2 months",
               "this week", "last week", "next week", "this month", "last month", "next month",
               "this quarter", "last quarter", "next quarter", "this year", "last year", "next year",
               "current month", "2024", "1999", "2025-03-15", "2025-02", "march", "march 2025",
               "sep 2024", "sept", "q1", "q3 2025", "h2 2024", "3 weeks ago", "10 days ago",
               "2 months ago", "the last 30 days", "in march", "last 0 days", "garbage", "13 days",
               "2025-13-01", "2023-02-29", "last quarter", "last 1 day", "september 10 2026",
               "september 10, 2026", "sep 10", "10 september 2026", "10th of sep", "2026/09/10",
               "this summer", "last winter", "this winter", "spring 2025", "next spring", "summer",
               "fall", "last autumn", "31 february", "feb 30 2025"]
    time_cases = []
    for t in [today, date(2024, 2, 29), date(2025, 12, 31), date(2026, 1, 5)]:
        for p in phrases:
            time_cases.append({"now": t.isoformat(), "text": p, "range": timeres.resolve(p, t)})
    (FIXTURES / "time.json").write_text(json.dumps(time_cases) + "\n")

    match_cases = []
    for ex in dataset("train", 30, 5) + dataset("eval", 10, 6):
        toks = match.model_tokens(ex.text)
        fs = match.match_spans(toks, match.field_entries(ex.schema))
        es = match.match_spans(toks, match.enum_entries(ex.schema), enum=True)
        match_cases.append({
            "text": ex.text, "schema": ex.schema,
            "fields": [[s.start, s.end, s.field, s.quality, int(s.alias)] for s in fs],
            "enums": [[s.start, s.end, s.field, s.value, list(s.owners)] for s in es],
        })
    (FIXTURES / "match.json").write_text(json.dumps(match_cases, ensure_ascii=False) + "\n")
    print(f"fixtures: {len(cases)} feature cases, {len(time_cases)} time cases, {len(match_cases)} match cases")


if __name__ == "__main__":
    import sys

    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 160000
    write_fixtures()
    write_eval_sets()
    build(samples, seed=0)
