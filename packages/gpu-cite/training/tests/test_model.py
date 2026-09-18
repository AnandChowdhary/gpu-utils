import torch

from gpu_cite.features import WIDTH
from gpu_cite.labels import ROLES, TAGS
from gpu_cite.model import CiteTagger, affine_scan, constrained_transitions, count_params, crf_nll, viterbi


def _sequential(a: torch.Tensor, b: torch.Tensor, reverse: bool) -> torch.Tensor:
    T = a.shape[1]
    h = torch.zeros_like(a[:, 0])
    out = torch.zeros_like(a)
    order = range(T - 1, -1, -1) if reverse else range(T)
    for t in order:
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out


def test_parallel_scan_matches_sequential_recurrence() -> None:
    torch.manual_seed(0)
    a = torch.rand(3, 37, 5)
    b = torch.randn(3, 37, 5)
    mask = torch.ones(3, 37, 1)
    mask[1, 20:] = 0.0
    for reverse in (False, True):
        got = affine_scan(a, b, mask, reverse)
        want = _sequential(a * mask + (1 - mask), b * mask, reverse) * mask
        assert torch.allclose(got, want, atol=1e-5), reverse


def test_forward_shapes_and_param_budget() -> None:
    model = CiteTagger()
    rows = torch.randint(1, 100, (2, 11, WIDTH))
    mask = torch.ones(2, 11)
    mask[1, 6:] = 0.0
    tags, parts, ty = model(rows, mask)
    assert tags.shape == (2, 11, len(TAGS))
    assert parts.shape == (2, 11, 3)
    assert ty.shape == (2, 8)
    n = count_params(model)
    assert 40_000 <= n <= 100_000, n


def test_padding_does_not_change_logits() -> None:
    torch.manual_seed(1)
    model = CiteTagger()
    rows = torch.randint(1, 100, (1, 9, WIDTH))
    tags_a, parts_a, ty_a = model(rows, torch.ones(1, 9))
    padded = torch.cat([rows, torch.zeros(1, 6, WIDTH, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 9), torch.zeros(1, 6)], dim=1)
    tags_b, parts_b, ty_b = model(padded, mask)
    assert torch.allclose(tags_a, tags_b[:, :9], atol=1e-5)
    assert torch.allclose(parts_a, parts_b[:, :9], atol=1e-5)
    assert torch.allclose(ty_a, ty_b, atol=1e-5)


def test_crf_and_viterbi_respect_bio_constraints() -> None:
    K = len(TAGS)
    R = len(ROLES)
    em = torch.zeros(4, K)
    em[0, 0] = 5.0
    em[1, 1 + R + 1] = 5.0  # I-TITLE strongly preferred but illegal after O
    em[2, 2] = 5.0  # B-TITLE
    em[3, 1 + R + 1] = 5.0  # I-TITLE legal after B-TITLE
    trans = constrained_transitions(torch.zeros(K, K))
    path = viterbi(em, trans)
    assert path[1] != 1 + R + 1
    assert path[2:] == [2, 1 + R + 1]
    nll = crf_nll(em.unsqueeze(0), torch.zeros(K, K), torch.tensor([path]), torch.ones(1, 4))
    assert torch.isfinite(nll) and nll.item() >= 0.0
