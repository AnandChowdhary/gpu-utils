"""gpu-paste model: summed sparse embeddings → bidirectional gated affine scan →
pointwise mix → mean-pooled context → BIO span head + kind head.

The TypeScript reference (src/cpu.ts) and the WGSL kernels mirror this file exactly:

  x_t      = sum_f E[id_f]                                   (D)
  pre_t    = W_dir x_t + b_dir                               (2D) per direction
  a_t      = sigmoid(pre_t[:D]),  u_t = tanh(pre_t[D:])
  h_t      = a_t * h_{t-1} + (1 - a_t) * u_t                 (forward and backward scans)
  m_t      = relu(W_mix [x_t; hf_t; hb_t] + b_mix)           (M)
  g        = mean_t m_t                                      (M)
  s_t      = relu(W_head [m_t; g] + b_head)                  (M)
  span_t   = W_out s_t + b_out                               (L = 1 + 2 * span kinds)
  kind     = W_k2 relu(W_k1 g + b_k1) + b_k2                 (K)

Quantization-aware training: every parameter is passed through a straight-through int6
fake-quantizer on each forward, matching gpu_utils_training.quant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gpu_paste.features import FEATURE_COUNT, TOTAL_ROWS

DIM = 32
MIX = 48
KIND_HIDDEN = 32
LEVELS = 31


def fake_quant(w: Tensor) -> Tensor:
    """Symmetric per-tensor int6 round trip with a straight-through gradient."""
    scale = w.detach().abs().max() / LEVELS
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(w / scale), -LEVELS, LEVELS) * scale
    return w + (q - w).detach()


def affine_scan(a: Tensor, b: Tensor) -> Tensor:
    """h_t = a_t * h_{t-1} + b_t over dim 1 as a Hillis-Steele parallel prefix (log T steps)."""
    n = a.shape[1]
    k = 1
    while k < n:
        a_prev = F.pad(a[:, :-k], (0, 0, k, 0), value=1.0)
        b_prev = F.pad(b[:, :-k], (0, 0, k, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        k *= 2
    return b


class PasteModel(nn.Module):
    def __init__(
        self,
        n_span_labels: int,
        n_kinds: int,
        rows: int = TOTAL_ROWS,
        dim: int = DIM,
        mix: int = MIX,
        kind_hidden: int = KIND_HIDDEN,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(rows, dim)
        nn.init.zeros_(self.embed.weight)
        self.gate_f = nn.Linear(dim, 2 * dim)
        self.gate_b = nn.Linear(dim, 2 * dim)
        self.mix = nn.Linear(3 * dim, mix)
        self.head = nn.Linear(2 * mix, mix)
        self.out = nn.Linear(mix, n_span_labels)
        self.kind1 = nn.Linear(mix, kind_hidden)
        self.kind2 = nn.Linear(kind_hidden, n_kinds)
        self.quantize = True

    def q(self, w: Tensor) -> Tensor:
        return fake_quant(w) if self.quantize else w

    def linear(self, layer: nn.Linear, x: Tensor) -> Tensor:
        return F.linear(x, self.q(layer.weight), self.q(layer.bias))

    def scan_direction(self, layer: nn.Linear, x: Tensor, mask: Tensor, backward: bool) -> Tensor:
        pre = self.linear(layer, x)
        a = torch.sigmoid(pre[..., : self.dim])
        u = torch.tanh(pre[..., self.dim :])
        b = (1.0 - a) * u
        # Padded positions become identity maps so they never leak into valid ones.
        m = mask.unsqueeze(-1)
        a = torch.where(m, a, torch.ones_like(a))
        b = torch.where(m, b, torch.zeros_like(b))
        if backward:
            a = a.flip(1)
            b = b.flip(1)
        h = affine_scan(a, b)
        return h.flip(1) if backward else h

    def forward(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """ids: [B, T, FEATURE_COUNT] int64, mask: [B, T] bool → (span logits [B,T,L], kind logits [B,K])."""
        assert ids.shape[-1] == FEATURE_COUNT
        x = F.embedding(ids, self.q(self.embed.weight)).sum(dim=2)
        hf = self.scan_direction(self.gate_f, x, mask, backward=False)
        hb = self.scan_direction(self.gate_b, x, mask, backward=True)
        m = torch.relu(self.linear(self.mix, torch.cat([x, hf, hb], dim=-1)))
        mf = mask.unsqueeze(-1).to(m.dtype)
        g = (m * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        gx = g.unsqueeze(1).expand(-1, m.shape[1], -1)
        s = torch.relu(self.linear(self.head, torch.cat([m, gx], dim=-1)))
        span = self.linear(self.out, s)
        kind = self.linear(self.kind2, torch.relu(self.linear(self.kind1, g)))
        return span, kind

    def export_tensors(self) -> dict[str, Tensor]:
        """Float tensors in the order the runtime expects (see src/cpu.ts)."""
        return {
            "embed": self.embed.weight.detach(),
            "gate_f.w": self.gate_f.weight.detach(),
            "gate_f.b": self.gate_f.bias.detach(),
            "gate_b.w": self.gate_b.weight.detach(),
            "gate_b.b": self.gate_b.bias.detach(),
            "mix.w": self.mix.weight.detach(),
            "mix.b": self.mix.bias.detach(),
            "head.w": self.head.weight.detach(),
            "head.b": self.head.bias.detach(),
            "out.w": self.out.weight.detach(),
            "out.b": self.out.bias.detach(),
            "kind1.w": self.kind1.weight.detach(),
            "kind1.b": self.kind1.bias.detach(),
            "kind2.w": self.kind2.weight.detach(),
            "kind2.b": self.kind2.bias.detach(),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
