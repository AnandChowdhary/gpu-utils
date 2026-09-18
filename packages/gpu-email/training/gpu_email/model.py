"""Dilated-CNN two-head tagger. Mirrors src/cpu.ts exactly.

  x0 = sum over slots of E[ids]                      [T, D]
  x_{l+1} = x_l + relu(conv1d_{k=3, dil=d_l}(x_l))     residual, zero padded
  h = relu(W_h x_L + b_h)                              [T, H]
  line logits = W_k h + b_k                            [T, 8]
  bio logits  = W_b h + b_b                            [T, 15]

Quantization-aware training: every parameter tensor is fake-quantized to int6 with a
per-tensor symmetric scale (straight-through estimator) once `qat` is on, matching
gpu_utils_training.quant.quantize at export time.
"""

from __future__ import annotations

import torch
from torch import nn

from gpu_email.features import BIO_LABELS, LINE_KINDS, NUM_ROWS, NUM_SLOTS

LEVELS = 31
DILATIONS = [1, 2, 4, 8, 16, 32]


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    scale = w.detach().abs().max() / LEVELS
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(w / scale), -LEVELS, LEVELS) * scale
    return w + (q - w).detach()


class EmailTagger(nn.Module):
    def __init__(self, dim: int = 48, hidden: int = 64, dilations: list[int] | None = None):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.dilations = dilations or DILATIONS
        self.emb = nn.Embedding(NUM_ROWS, dim)
        nn.init.normal_(self.emb.weight, std=0.15)
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 3, padding=d, dilation=d) for d in self.dilations]
        )
        self.head = nn.Linear(dim, hidden)
        self.kind_out = nn.Linear(hidden, len(LINE_KINDS))
        self.bio_out = nn.Linear(hidden, len(BIO_LABELS))
        self.qat = False

    def _w(self, w: torch.Tensor) -> torch.Tensor:
        return fake_quant(w) if self.qat else w

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids: [B, T, NUM_SLOTS] long; mask: [B, T] float (1 = real token)."""
        assert ids.shape[-1] == NUM_SLOTS
        m = mask.unsqueeze(-1)
        emb = nn.functional.embedding(ids, self._w(self.emb.weight))  # [B,T,S,D]
        x = emb.sum(dim=2) * m
        xt = x.transpose(1, 2)  # [B, D, T]
        mt = m.transpose(1, 2)
        for conv in self.convs:
            y = nn.functional.conv1d(xt, self._w(conv.weight), self._w(conv.bias), padding=conv.padding, dilation=conv.dilation)
            xt = (xt + torch.relu(y)) * mt
        x = xt.transpose(1, 2)
        h = torch.relu(nn.functional.linear(x, self._w(self.head.weight), self._w(self.head.bias)))
        kind = nn.functional.linear(h, self._w(self.kind_out.weight), self._w(self.kind_out.bias))
        bio = nn.functional.linear(h, self._w(self.bio_out.weight), self._w(self.bio_out.bias))
        return kind, bio

    def export_tensors(self) -> dict[str, torch.Tensor]:
        """Tensors in the layout src/cpu.ts reads: conv weights as [k, in, out]."""
        out: dict[str, torch.Tensor] = {"emb": self.emb.weight.detach()}
        for i, conv in enumerate(self.convs):
            # torch conv1d weight is [out, in, k]; store as [k, in, out] for row-major reads
            out[f"conv{i}.w"] = conv.weight.detach().permute(2, 1, 0).contiguous()
            out[f"conv{i}.b"] = conv.bias.detach()
        out["head.w"] = self.head.weight.detach().t().contiguous()  # [D, H]
        out["head.b"] = self.head.bias.detach()
        out["kind.w"] = self.kind_out.weight.detach().t().contiguous()  # [H, 8]
        out["kind.b"] = self.kind_out.bias.detach()
        out["bio.w"] = self.bio_out.weight.detach().t().contiguous()  # [H, 15]
        out["bio.b"] = self.bio_out.bias.detach()
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
