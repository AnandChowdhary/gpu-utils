"""Synthetic dataset: generation, markup I/O, token tagging and caching.

`uv run python -m gpu_log.data` builds and caches the default train/held-out sets and
prints a few annotated samples.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gpu_utils_training.batch import pad_labels, to_torch, pack_rows

from .features import FEATURE_COUNT, featurize
from .formats import ENTRY_BUILDERS, MULTI_BUILDERS
from .gen import KINDS, ROLES, Line
from .traces import TRACE_BUILDERS

LABELS = ["O"] + [f"{p}-{r}" for r in ROLES for p in ("B", "I")]
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class Example:
    text: str
    spans: tuple[tuple[int, int, str], ...]
    kind: str


def from_line(line: Line) -> Example:
    return Example(line.text(), tuple(line.spans()), line.kind)


MARKUP_RE = re.compile(r"⟦([A-Z]+)\|(.*?)⟧", re.S)


def parse_markup(s: str) -> Example:
    """'⟦TS|...⟧ text ⟦MSG|...⟧ ##kind' -> Example."""
    body, _, kind = s.rpartition(" ##")
    kind = kind.strip()
    if kind not in KIND_INDEX:
        raise ValueError(f"bad kind in: {s!r}")
    text: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    n = 0
    for m in MARKUP_RE.finditer(body):
        text.append(body[pos:m.start()])
        n += len(body[pos:m.start()])
        role, inner = m.group(1), m.group(2)
        if role not in ROLES:
            raise ValueError(f"bad role {role} in: {s!r}")
        spans.append((n, n + len(inner), role))
        text.append(inner)
        n += len(inner)
        pos = m.end()
    text.append(body[pos:])
    return Example("".join(text), tuple(spans), kind)


def to_markup(ex: Example) -> str:
    out = []
    pos = 0
    for s, e, r in ex.spans:
        out.append(ex.text[pos:s])
        out.append(f"⟦{r}|{ex.text[s:e]}⟧")
        pos = e
    out.append(ex.text[pos:])
    return "".join(out) + f" ##{ex.kind}"


def read_jsonl(path: Path) -> list[Example]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            d = json.loads(raw)
            out.append(Example(d["text"], tuple(tuple(s) for s in d["spans"]), d["kind"]))
    return out


def read_markup_file(path: Path) -> list[Example]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#!"):
            continue
        out.append(parse_markup(raw))
    return out


def add_noise(rng: random.Random, ex: Example) -> Example:
    """Trailing whitespace, rare truncation, rare leading indent on entries."""
    text, spans, kind = ex.text, list(ex.spans), ex.kind
    if rng.random() < 0.05:
        text = text + rng.choice([" ", "  ", "\t", " \t"])
    if rng.random() < 0.03 and len(text) > 40:
        cut = rng.randint(20, len(text) - 1)
        while cut > 0 and text[cut] != " ":
            cut -= 1
        if cut > 0:
            text = text[:cut]
            spans = [(s, min(e, cut), r) for s, e, r in spans if s < cut]
    if kind == "entry" and rng.random() < 0.02:
        pad = rng.choice([" ", "  "])
        text = pad + text
        spans = [(s + len(pad), e + len(pad), r) for s, e, r in spans]
    if rng.random() < 0.02:
        # very long message tail: repeated words
        extra = " " + " ".join(rng.choice(["lorem", "ipsum", "dolor", "sit", "amet", "x", "0", "-", "..."]) for _ in range(rng.randint(40, 200)))
        # attach to the last MSG span if one ends the line, else plain O text
        if spans and spans[-1][2] == "MSG" and spans[-1][1] == len(text):
            s, e, r = spans[-1]
            spans[-1] = (s, e + len(extra), r)
        text = text + extra
    return Example(text, tuple(spans), kind)


def generate(n_lines: int, seed: int) -> list[Example]:
    """Roughly n_lines examples: ~57% entries, the rest trace frames and continuations."""
    rng = random.Random(seed)
    entry_builders = [b for b, _ in ENTRY_BUILDERS]
    entry_weights = [w for _, w in ENTRY_BUILDERS]
    multi = [b for b, _ in MULTI_BUILDERS]
    multi_w = [w for _, w in MULTI_BUILDERS]
    trace = [b for b, _ in TRACE_BUILDERS]
    trace_w = [w for _, w in TRACE_BUILDERS]
    out: list[Example] = []
    while len(out) < n_lines:
        v = rng.random()
        if v < 0.90:
            lines = [rng.choices(entry_builders, entry_weights)[0](rng)]
        elif v < 0.92:
            lines = rng.choices(multi, multi_w)[0](rng)
        else:
            lines = rng.choices(trace, trace_w)[0](rng)
        for line in lines:
            out.append(add_noise(rng, from_line(line)))
    return out[:n_lines]


def tag_example(ex: Example) -> tuple[list[list[int]], list[int], int, int]:
    """Returns (feature rows, tag ids, kind id, misaligned token count)."""
    tokens, rows = featurize(ex.text)
    tags = [0] * len(tokens)
    # spans are code-point offsets; tokens carry UTF-16 offsets (to match JavaScript)
    u16 = [0]
    for ch in ex.text:
        u16.append(u16[-1] + (2 if ord(ch) > 0xFFFF else 1))
    spans = sorted((u16[s], u16[e], r) for s, e, r in ex.spans)
    si = 0
    misaligned = 0
    for i, t in enumerate(tokens):
        while si < len(spans) and spans[si][1] <= t.start:
            si += 1
        if si < len(spans):
            s, e, role = spans[si]
            if s <= t.start < e:
                if t.end > e:
                    misaligned += 1
                tags[i] = LABEL_INDEX[("B-" if t.start == s else "I-") + role]
    return rows, tags, KIND_INDEX[ex.kind], misaligned


@dataclass
class Dataset:
    features: np.ndarray  # [T, F] int32
    tags: np.ndarray  # [T] int8
    offsets: np.ndarray  # [L + 1] int64 token offsets per line
    kinds: np.ndarray  # [L] int8

    def __len__(self) -> int:
        return len(self.kinds)

    def line(self, i: int) -> tuple[np.ndarray, np.ndarray, int]:
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.features[s:e], self.tags[s:e], int(self.kinds[i])


def build(examples: list[Example]) -> tuple[Dataset, int]:
    feats: list[list[int]] = []
    tags: list[int] = []
    offsets = [0]
    kinds = []
    misaligned = 0
    for ex in examples:
        rows, t, k, mis = tag_example(ex)
        feats.extend(rows)
        tags.extend(t)
        offsets.append(len(tags))
        kinds.append(k)
        misaligned += mis
    return (
        Dataset(
            np.asarray(feats, dtype=np.int32).reshape(-1, FEATURE_COUNT),
            np.asarray(tags, dtype=np.int8),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(kinds, dtype=np.int8),
        ),
        misaligned,
    )


def cached(name: str, n_lines: int, seed: int) -> Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}-{n_lines}-{seed}.npz"
    if path.exists():
        z = np.load(path)
        return Dataset(z["features"], z["tags"], z["offsets"], z["kinds"])
    ds, mis = build(generate(n_lines, seed))
    np.savez_compressed(path, features=ds.features, tags=ds.tags, offsets=ds.offsets, kinds=ds.kinds)
    print(f"built {name}: {len(ds)} lines, {len(ds.tags)} tokens, {mis} misaligned tokens -> {path}")
    return ds


def main() -> None:
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for ex in generate(n, 7):
        print(to_markup(ex))
    cached("train", 160_000, 1)
    cached("heldout", 12_000, 2)


if __name__ == "__main__":
    main()


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def batches(ds: Dataset, batch_lines: int, rng: np.random.Generator, padding_id: int, slots: int = FEATURE_COUNT) -> list[Batch]:
    """Length-bucketed batches: (rows [B, T, slots] int64, mask [B, T] bool, tags [B, T] (-100 padded), kinds [B])."""
    n = len(ds)
    lengths = np.diff(ds.offsets)
    order = rng.permutation(n)
    chunk = batch_lines * 50
    groups: list[np.ndarray] = []
    for c in range(0, n, chunk):
        idx = order[c : c + chunk]
        idx = idx[np.argsort(lengths[idx], kind="stable")]
        for b in range(0, len(idx), batch_lines):
            groups.append(idx[b : b + batch_lines])
    rng.shuffle(groups)  # type: ignore[arg-type]
    out: list[Batch] = []
    for idx in groups:
        seqs = [ds.features[ds.offsets[i] : ds.offsets[i + 1]].tolist() for i in idx]
        rows, lens = pack_rows(seqs, slots, padding_id)
        r, m = to_torch(rows, lens)
        tags = pad_labels([ds.tags[ds.offsets[i] : ds.offsets[i + 1]].astype(np.int64).tolist() for i in idx], rows.shape[1])
        out.append((r, m, tags, torch.from_numpy(ds.kinds[idx].astype(np.int64))))
    return out


def loss(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    rows, mask, tags, kinds = batch
    out = model(rows, mask)
    tl = F.cross_entropy(out["tags"].reshape(-1, out["tags"].shape[-1]), tags.reshape(-1), ignore_index=-100)
    kl = F.cross_entropy(out["pooled"], kinds)
    return tl + 0.5 * kl
