import torch

from gpu_paste.data import LABELS, LEARNED_KINDS
from gpu_paste.model import PasteModel, affine_scan, parameter_count


def test_affine_scan_matches_sequential() -> None:
    torch.manual_seed(0)
    a = torch.rand(2, 37, 4)
    b = torch.randn(2, 37, 4)
    h = affine_scan(a, b)
    ref = torch.zeros(2, 4)
    for t in range(37):
        ref = a[:, t] * ref + b[:, t]
        assert torch.allclose(h[:, t], ref, atol=1e-5)


def test_parameter_budget() -> None:
    m = PasteModel(n_span_labels=len(LABELS), n_kinds=len(LEARNED_KINDS))
    n = parameter_count(m)
    assert 40_000 <= n <= 80_000, n
    ids = torch.zeros(3, 9, 10, dtype=torch.int64)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[1, 5:] = False
    span, kind = m(ids, mask)
    assert span.shape == (3, 9, len(LABELS))
    assert kind.shape == (3, len(LEARNED_KINDS))
