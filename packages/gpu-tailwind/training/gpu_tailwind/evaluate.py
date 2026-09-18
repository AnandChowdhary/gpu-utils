"""Tag-level evaluation of the promoted checkpoint on the held-out generated set.

    uv run python -m gpu_tailwind.evaluate

Class-level metrics (exact-set match, per-class precision/recall) need the TypeScript
compiler; run `pnpm --filter gpu-tailwind eval` for those (scripts/eval.ts).
"""

from __future__ import annotations

import json
from collections import Counter

import torch

from .data import LABELS
from .export import dequantized
from .model import OUT, Tagger
from .train import RUNS, batches, encode, load


def spans(labels: list[int]) -> set[tuple[int, int, str]]:
    out = set()
    start = None
    kind = None
    for i, l in enumerate(labels + [0]):
        name = LABELS[l] if l >= 0 and l < len(LABELS) else "O"
        if (
            name.startswith("B-")
            or name in ("O", "SEP", "NEG")
            or (name.startswith("I-") and name[2:] != kind)
        ):
            if start is not None:
                out.add((start, i, kind))  # type: ignore[arg-type]
                start = None
            if name.startswith("B-") or name.startswith("I-"):
                start, kind = i, name[2:]
    return out


def main() -> None:
    torch.set_num_threads(2)
    model = Tagger()
    model.load_state_dict(torch.load(RUNS / "best.pt"))
    model = dequantized(model)
    data = encode(load("heldout"))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    tok_ok = tok_n = seq_ok = seq_n = b_ok = b_n = 0
    with torch.no_grad():
        for feats, labels, bound, mask in batches(
            data, 256, __import__("random").Random(0), False
        ):
            out = model(feats, mask)
            pred = out[..., : OUT - 1].argmax(-1)
            for i in range(feats.shape[0]):
                n = int(mask[i].sum())
                p = pred[i, :n].tolist()
                g = labels[i, :n].tolist()
                tok_ok += sum(int(a == b) for a, b in zip(p, g))
                tok_n += n
                seq_ok += int(p == g)
                seq_n += 1
                bp = (out[i, :n, OUT - 1] > 0).long().tolist()
                bg = bound[i, :n].long().tolist()
                b_ok += sum(int(a == b) for a, b in zip(bp, bg))
                b_n += n
                ps, gs = spans(p), spans(g)
                for s in ps:
                    (tp if s in gs else fp)[s[2]] += 1
                for s in gs:
                    if s not in ps:
                        fn[s[2]] += 1
    result: dict[str, float] = {
        "n": seq_n,
        "token_acc": tok_ok / tok_n,
        "seq_acc": seq_ok / seq_n,
        "boundary_acc": b_ok / b_n,
    }
    print(f"held-out: {seq_n} examples")
    print(f"token accuracy   {tok_ok / tok_n:.4f}")
    print(f"sequence exact   {seq_ok / seq_n:.4f}")
    print(f"boundary accuracy {b_ok / b_n:.4f}")
    for kind in ("PROP", "VAL", "VAR"):
        p = tp[kind] / max(1, tp[kind] + fp[kind])
        r = tp[kind] / max(1, tp[kind] + fn[kind])
        f1 = 2 * p * r / max(1e-9, p + r)
        result[f"{kind}_precision"], result[f"{kind}_recall"], result[f"{kind}_f1"] = p, r, f1
        print(f"span {kind:5s} precision {p:.4f} recall {r:.4f} f1 {f1:.4f}")
    (RUNS / "tag_eval.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
