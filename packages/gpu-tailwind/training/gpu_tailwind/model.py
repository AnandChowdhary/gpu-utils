"""Bidirectional gated affine-scan tagger with int6 quantization-aware training.

e_t   = sum of 7 sparse embeddings                      [D]
a,u   = sigmoid(Wa e + ba), tanh(Wu e + bu)  per direction
h_t   = a * h_{t-1} + (1 - a) * u             (forward and backward scans)
x_t   = [e_t ; h_f,t ; h_b,t]                            [3D]
c_t   = depthwise conv3(x)                               [3D]
g     = mean_t x_t                                       [3D]
z_t   = relu(W1 c_t + Wg g + b1)                         [H]
out_t = W2 z_t + b2                                      [9 roles + 1 boundary]

src/cpu.ts and src/shader.wgsl implement exactly this graph.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .data import LABELS
from .features import ROWS, WIDTH

D = 24
H = 32
OUT = len(LABELS) + 1
LEVELS = 31


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor) -> Tensor:  # type: ignore[override]
        scale = w.detach().abs().max() / LEVELS
        scale = torch.clamp(scale, min=1e-8)
        return torch.clamp(torch.round(w / scale), -LEVELS, LEVELS) * scale

    @staticmethod
    def backward(ctx, g: Tensor) -> Tensor:  # type: ignore[override]
        return g


def fq(w: Tensor, enabled: bool) -> Tensor:
    return FakeQuant.apply(w) if enabled else w  # type: ignore[no-any-return]


class Tagger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Parameter(torch.randn(ROWS, D) * 0.1)
        self.wa_f = nn.Parameter(torch.randn(D, D) * 0.15)
        self.ba_f = nn.Parameter(torch.zeros(D))
        self.wu_f = nn.Parameter(torch.randn(D, D) * 0.15)
        self.bu_f = nn.Parameter(torch.zeros(D))
        self.wa_b = nn.Parameter(torch.randn(D, D) * 0.15)
        self.ba_b = nn.Parameter(torch.zeros(D))
        self.wu_b = nn.Parameter(torch.randn(D, D) * 0.15)
        self.bu_b = nn.Parameter(torch.zeros(D))
        self.conv = nn.Parameter(torch.randn(3, 3 * D) * 0.2)
        self.bc = nn.Parameter(torch.zeros(3 * D))
        self.w1 = nn.Parameter(torch.randn(3 * D, H) * 0.12)
        self.wg = nn.Parameter(torch.randn(3 * D, H) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.w2 = nn.Parameter(torch.randn(H, OUT) * 0.15)
        self.b2 = nn.Parameter(torch.zeros(OUT))
        self.quant = False

    def tensors(self) -> dict[str, Tensor]:
        return {n: p.detach() for n, p in self.named_parameters()}

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        """feats: [B, T, WIDTH] int64 ids; mask: [B, T] bool. Returns [B, T, OUT]."""
        q = self.quant
        emb = fq(self.emb, q)
        e = emb[feats].sum(2)  # [B, T, D]
        m = mask.unsqueeze(-1).to(e.dtype)
        e = e * m
        bsz, t, _ = e.shape

        def scan(wa: Tensor, ba: Tensor, wu: Tensor, bu: Tensor, reverse: bool) -> Tensor:
            a = torch.sigmoid(e @ fq(wa, q) + fq(ba, q))
            u = torch.tanh(e @ fq(wu, q) + fq(bu, q))
            b = (1 - a) * u
            hs = []
            h = torch.zeros(bsz, D, dtype=e.dtype)
            idx = range(t - 1, -1, -1) if reverse else range(t)
            for i in idx:
                h = a[:, i] * h + b[:, i]
                hs.append(h)
            if reverse:
                hs.reverse()
            return torch.stack(hs, 1)

        hf = scan(self.wa_f, self.ba_f, self.wu_f, self.bu_f, False)
        hb = scan(self.wa_b, self.ba_b, self.wu_b, self.bu_b, True)
        x = torch.cat([e, hf, hb], -1) * m  # [B, T, 3D]
        conv = fq(self.conv, q)
        xp = torch.nn.functional.pad(x, (0, 0, 1, 1))
        c = xp[:, :-2] * conv[0] + xp[:, 1:-1] * conv[1] + xp[:, 2:] * conv[2] + fq(self.bc, q)
        g = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).to(e.dtype)  # [B, 3D]
        z = torch.relu(c @ fq(self.w1, q) + (g @ fq(self.wg, q)).unsqueeze(1) + fq(self.b1, q))
        return z @ fq(self.w2, q) + fq(self.b2, q)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["Tagger", "D", "H", "OUT", "WIDTH", "count_params"]
