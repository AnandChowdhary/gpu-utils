import numpy as np
import torch

from gpu_log.features import featurize
from gpu_log.model import LogTagger, dequantized, forward_numpy


def test_numpy_forward_matches_torch_with_line_masking() -> None:
    torch.manual_seed(0)
    model = LogTagger()
    model.set_quant(True)
    model.eval()
    texts = ["2024-01-15 INFO hello world k=v", "\tat a.b.C.d(C.java:12)"]
    deq = dequantized(model.tensors())
    rows = [np.asarray(featurize(t)[1], dtype=np.int64) for t in texts]
    L = max(len(r) for r in rows)
    feats = np.zeros((2, L, rows[0].shape[1]), dtype=np.int64)
    mask = np.zeros((2, L), dtype=np.float32)
    for i, r in enumerate(rows):
        feats[i, : len(r)] = r
        mask[i, : len(r)] = 1
    with torch.no_grad():
        tags, kinds = model(torch.from_numpy(feats), torch.from_numpy(mask))
    for i, r in enumerate(rows):
        ref = forward_numpy(deq, r)
        assert np.abs(tags[i, : len(r)].numpy() - ref[:, :23]).max() < 1e-4
        assert np.abs(kinds[i].numpy() - ref[:, 23:].mean(0)).max() < 1e-4


def test_param_budget() -> None:
    n = LogTagger().param_count()
    assert 100_000 <= n <= 250_000
