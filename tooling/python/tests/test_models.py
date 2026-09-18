import numpy as np
import torch

from gpu_utils_training.batch import collate
from gpu_utils_training.kernels import conv_tensor_names, scan_tensor_names
from gpu_utils_training.models import ConvTagger, ScanTagger, from_config


def models():
    torch.manual_seed(0)
    return [ScanTagger(50, 8, 4, pooled_out=2, scan_layers=2), ConvTagger(50, 6, 8, 3, [1, 2, 4], 4, pooled_out=2), ScanTagger(50, 8, 4)]


def test_interface_shapes() -> None:
    for m in models():
        rows, mask = collate([[[1, 2], [3]], [[4]]], 2, m.padding_id)
        out = m(rows, mask)
        assert out["tags"].shape == (2, 2, 4)
        if m.pooled_out:
            assert out["pooled"].shape == (2, 2)
        else:
            assert out["pooled"] is None


def test_batch_invariance() -> None:
    rng = np.random.default_rng(1)
    for m in models():
        seqs = [[[int(v) for v in rng.integers(0, 50, size=rng.integers(1, 3))] for _ in range(rng.integers(1, 9))] for _ in range(5)]
        rows, mask = collate(seqs, 2, m.padding_id)
        out = m(rows, mask)
        for i, s in enumerate(seqs):
            r1, m1 = collate([s], 2, m.padding_id)
            one = m(r1, m1)
            assert torch.allclose(one["tags"][0], out["tags"][i, : len(s)], atol=1e-5)
            if m.pooled_out:
                assert torch.allclose(one["pooled"][0], out["pooled"][i], atol=1e-5)


def test_tensor_order_matches_kernel_tables() -> None:
    for m in models():
        cfg = m.config()
        names = scan_tensor_names(cfg) if m.family == "scan" else conv_tensor_names(cfg)
        assert m.tensor_names() == names
        shapes = {k: v.shape for k, v in m.tensors().items()}
        assert shapes["embed"] == (50, m.embed_dim)
        assert shapes["tags.w"] == (m.head_dim, 4)


def test_from_config_round_trip() -> None:
    for m in models():
        clone = from_config(m.config())
        clone.load_tensors(m.tensors())
        rows, mask = collate([[[1, 2], [3], [7]]], 2, m.padding_id)
        assert torch.allclose(clone(rows, mask)["tags"], m(rows, mask)["tags"])
        assert clone.config() == m.config()
