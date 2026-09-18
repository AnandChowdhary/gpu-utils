import torch
from torch.nn import functional as F

from gpu_utils_training.layers import (
    BiScan,
    DepthwiseConv,
    DilatedResidualBlock,
    affine_scan,
    masked_mean,
    sparse_embed,
)


def sequential_scan(a: torch.Tensor, b: torch.Tensor, reverse: bool) -> torch.Tensor:
    out = torch.zeros_like(b)
    h = torch.zeros_like(b[:, 0])
    order = range(a.shape[1] - 1, -1, -1) if reverse else range(a.shape[1])
    for t in order:
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out


def test_affine_scan_matches_sequential() -> None:
    torch.manual_seed(0)
    a, b = torch.rand(2, 11, 3), torch.randn(2, 11, 3)
    for reverse in (False, True):
        assert torch.allclose(affine_scan(a, b, reverse=reverse), sequential_scan(a, b, reverse), atol=1e-6)


def test_affine_scan_padding_is_identity() -> None:
    torch.manual_seed(1)
    a, b = torch.rand(1, 6, 2), torch.randn(1, 6, 2)
    mask = torch.tensor([[True, True, True, True, False, False]])
    full = affine_scan(a[:, :4], b[:, :4], reverse=True)
    padded = affine_scan(a, b, mask, reverse=True)
    assert torch.allclose(padded[:, :4], full, atol=1e-6)
    assert torch.equal(padded[:, 4:], torch.zeros(1, 2, 2))


def test_sparse_embed_skips_padding() -> None:
    table = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    rows = torch.tensor([[[0, 4, 4], [1, 2, 4]]])  # padding id = 4 (= rows in table)
    out = sparse_embed(rows, table, padding_id=4)
    assert torch.equal(out[0, 0], table[0])
    assert torch.equal(out[0, 1], table[1] + table[2])
    assert torch.equal(sparse_embed(torch.tensor([[[0, 1]]]), table, padding_id=0)[0, 0], table[1])


def test_depthwise_conv_matches_conv1d() -> None:
    torch.manual_seed(2)
    conv = DepthwiseConv(4, taps=5)
    x = torch.randn(2, 9, 4)
    ref = F.conv1d(x.transpose(1, 2), conv.w.w.t().unsqueeze(1), conv.b.w, padding=2, groups=4).transpose(1, 2)
    assert torch.allclose(conv(x), ref, atol=1e-6)


def test_dilated_block_matches_conv1d() -> None:
    torch.manual_seed(3)
    blk = DilatedResidualBlock(5, dilation=2)
    x = torch.randn(1, 10, 5)
    w1 = blk.w1.w.permute(2, 1, 0)  # [out, in, tap]
    h = F.relu(F.conv1d(x.transpose(1, 2), w1, blk.b1.w, padding=2, dilation=2)).transpose(1, 2)
    ref = x + h @ blk.w2.w + blk.b2.w
    assert torch.allclose(blk(x), ref, atol=1e-6)


def test_biscan_shapes_and_masking() -> None:
    torch.manual_seed(4)
    scan = BiScan(3, 4)
    x = torch.randn(2, 7, 3)
    mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
    out = scan(x, mask)
    assert out.shape == (2, 7, 8)
    solo = scan(x[1:, :3], torch.ones(1, 3, dtype=torch.bool))
    assert torch.allclose(out[1, :3], solo[0], atol=1e-6)
    assert torch.equal(out[1, 3:], torch.zeros(4, 8))
    assert set(scan.tensors("s")) == {f"s.{d}.{p}" for d in "fb" for p in ("wa", "ba", "wu", "bu")}


def test_masked_mean() -> None:
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(x, torch.tensor([[True, True, False]])).item() == 2.0
