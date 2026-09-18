import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gpu_utils_training.batch import collate, pad_labels
from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.export import check_fixtures, export_package
from gpu_utils_training.loop import load_checkpoint, train
from gpu_utils_training.models import ScanTagger, from_config
from gpu_utils_training.quant import decode_weights

SLOTS = 2


def toy_data(n: int, rng: np.random.Generator):
    """Tag = 1 when the first feature id is even; a scan tagger learns it in a few steps."""
    seqs, labels = [], []
    for _ in range(n):
        rows = [[int(v) for v in rng.integers(0, 20, size=SLOTS)] for _ in range(int(rng.integers(1, 6)))]
        seqs.append(rows)
        labels.append([r[0] % 2 for r in rows])
    return seqs, labels


def test_train_loop_writes_checkpoints(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    seqs, labels = toy_data(64, rng)
    model = ScanTagger(20, 4, 2)

    def make_batches(epoch: int, rng: np.random.Generator):
        order = rng.permutation(len(seqs))
        return [(collate([seqs[i] for i in idx], SLOTS, model.padding_id), pad_labels([labels[i] for i in idx], max(len(seqs[i]) for i in idx))) for idx in np.array_split(order, 4)]

    def loss(m, batch):
        (rows, mask), y = batch
        out = m(rows, mask)["tags"]
        return F.cross_entropy(out.reshape(-1, 2), y.reshape(-1), ignore_index=-100)

    seen = []

    def evaluate(m):
        seen.append(m.quant)
        rows, mask = collate(seqs, SLOTS, model.padding_id)
        pred = m(rows, mask)["tags"].argmax(-1)
        y = pad_labels(labels, rows.shape[1])
        return {"token_accuracy": float(((pred == y) & mask).sum() / mask.sum())}

    result = train(model, make_batches, evaluate, loss=loss, epochs=3, lr=1e-2, weight_decay=0.0, qat_from=1, threads=1, seed=0, out_dir=tmp_path, select="token_accuracy", warmup=2, log=lambda s: None)
    assert seen == [True, True, True]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history["history"]) == 3 and history["best"]["epoch"] in (0, 1, 2)
    assert result["best"]["token_accuracy"] == max(h["token_accuracy"] for h in history["history"])
    fresh = from_config(model.config())
    ckpt = load_checkpoint(fresh, tmp_path / "best.pt")
    assert ckpt["config"] == model.config()

    # Wall-clock budget: stops after the first step but still evaluates and saves.
    budget = train(ScanTagger(20, 4, 2), make_batches, evaluate, loss=loss, epochs=5, lr=1e-2, out_dir=tmp_path / "b", select="token_accuracy", minutes=0.0, log=lambda s: None)
    assert len(budget["history"]) == 1


def test_export_package_canonical_fixtures(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = ScanTagger(20, 4, 3, pooled_out=2)
    fixtures = [{"input": "a b", "rows": [[1, 2], [3]], "tokens": ["a", "b"]}, {"input": {"text": "x", "schema": 1}, "rows": [[5]]}, {"input": "", "rows": []}]
    manifest = export_package(model, tmp_path, ["O", "B-X", "I-X"], {"name": "gpu-test", "extra": 1}, fixtures)
    assert manifest["family"] == "scan" and manifest["slots"] == 2 and manifest["labels"] == ["O", "B-X", "I-X"] and manifest["extra"] == 1
    data = json.loads((tmp_path / "fixtures.json").read_text())
    assert set(data) == {"cases"}
    case = data["cases"][0]
    assert set(case) == {"input", "rows", "logits", "pooled", "tokens"}
    assert len(case["logits"]) == 2 and len(case["logits"][0]) == 3 and len(case["pooled"]) == 2
    assert data["cases"][1]["input"] == {"text": "x", "schema": 1}
    assert data["cases"][2]["logits"] == []
    # QAT-on model reproduces the fixtures (fake_quant == decoded int6 weights).
    assert check_fixtures(model, tmp_path) < 1e-6
    decoded = decode_weights((tmp_path / "weights.txt").read_text(), manifest)
    assert set(decoded) == set(model.tensor_names())


def test_cli_defaults(tmp_path: Path) -> None:
    args = training_parser("x", epochs=3).parse_args(["--seed", "5", "--minutes", "2"])
    assert args.epochs == 3 and args.seed == 5 and args.minutes == 2.0 and args.threads == 2 and args.run == "default"
    kw = loop_kwargs(args)
    assert kw["epochs"] == 3 and kw["seed"] == 5 and "lr" in kw
    anchor = tmp_path / "pkg" / "train.py"
    anchor.parent.mkdir()
    assert run_dir(anchor, args) == tmp_path / "runs" / "default"
