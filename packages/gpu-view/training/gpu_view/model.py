"""Bidirectional gated affine-scan tagger (gpu-time / gpu-query family).

Summed sparse embeddings → depthwise 5-tap convolution → gate/candidate →
forward and backward affine scans `h[t] = a[t] * h[t-1] + b[t]` (a parallel
prefix scan over affine maps) → combine → mean-pooled gated global context →
two-layer head emitting one logit per role plus one clause-boundary logit.
Mirrors src/cpu.ts and src/shader.wgsl exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HIDDEN = 32
HEAD_GATE = 16
HEAD_HIDDEN = 64
CONV = 5


def affine_scan(gate: Tensor, cand: Tensor) -> Tensor:
    """Inclusive scan of state[t] = gate[t] * state[t-1] + cand[t] (Hillis-Steele over affine maps)."""
    width = gate.shape[1]
    stride = 1
    while stride < width:
        g2 = gate[:, stride:] * gate[:, :-stride]
        c2 = cand[:, stride:] + gate[:, stride:] * cand[:, :-stride]
        gate = torch.cat((gate[:, :stride], g2), dim=1)
        cand = torch.cat((cand[:, :stride], c2), dim=1)
        stride *= 2
    return cand


def fake_quant(w: Tensor, bits: int = 6) -> Tensor:
    levels = (1 << (bits - 1)) - 1
    scale = (w.detach().abs().max() / levels).clamp_min(1e-8)
    q = (w / scale).round().clamp(-levels, levels) * scale
    return w + (q - w).detach()


class ViewTagger(nn.Module):
    def __init__(self, feature_rows: int, roles: int) -> None:
        super().__init__()
        self.feature_rows = feature_rows
        self.roles = roles
        self.embedding = nn.Parameter(torch.empty(feature_rows, HIDDEN))
        self.encoder_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.convolution = nn.Parameter(torch.empty(CONV, HIDDEN))
        self.gate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.gate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.candidate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.candidate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.combine_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN * 2))
        self.combine_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.global_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.global_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.head_gate_weight = nn.Parameter(torch.empty(HEAD_GATE, HIDDEN * 2))
        self.head_gate_bias = nn.Parameter(torch.zeros(HEAD_GATE))
        self.head_hidden_weight = nn.Parameter(torch.empty(HEAD_HIDDEN, HIDDEN * 2 + HEAD_GATE))
        self.head_hidden_bias = nn.Parameter(torch.zeros(HEAD_HIDDEN))
        self.output_weight = nn.Parameter(torch.empty(roles + 1, HEAD_HIDDEN))
        self.output_bias = nn.Parameter(torch.zeros(roles + 1))
        self.qat = False
        for _name, p in self.named_parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding, std=0.08)
        nn.init.normal_(self.convolution, std=0.15)
        with torch.no_grad():
            self.gate_bias.copy_(torch.linspace(0.0, 4.0, HIDDEN))

    # Export order; src/cpu.ts reads tensors by these names.
    TENSOR_NAMES = [
        "embedding", "encoder_bias", "convolution", "gate_weight", "gate_bias", "candidate_weight",
        "candidate_bias", "combine_weight", "combine_bias", "global_weight", "global_bias",
        "head_gate_weight", "head_gate_bias", "head_hidden_weight", "head_hidden_bias",
        "output_weight", "output_bias",
    ]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def w(self, name: str) -> Tensor:
        v = getattr(self, name)
        return fake_quant(v) if self.qat else v

    def linear(self, x: Tensor, name: str) -> Tensor:
        return F.linear(x, self.w(f"{name}_weight"), self.w(f"{name}_bias"))

    def forward(self, rows: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """rows: [B, T, SLOTS] int64 (padding row = feature_rows); valid: [B, T] bool."""
        mask = (rows != self.feature_rows).unsqueeze(-1)
        emb = F.embedding(rows.clamp_max(self.feature_rows - 1), self.w("embedding"))
        emb = (emb * mask).sum(dim=2)
        v = valid.unsqueeze(-1).to(emb.dtype)
        emb = emb * v

        kernel = self.w("convolution").transpose(0, 1).unsqueeze(1)  # [H, 1, 5]
        enc = F.conv1d(emb.transpose(1, 2), kernel, padding=CONV // 2, groups=HIDDEN).transpose(1, 2)
        enc = torch.tanh(enc + self.w("encoder_bias")) * v

        gate = torch.sigmoid(self.linear(enc, "gate"))
        cand = (1 - gate) * torch.tanh(self.linear(enc, "candidate"))
        gate = torch.where(valid.unsqueeze(-1), gate, torch.ones_like(gate))
        cand = cand * v
        fwd = affine_scan(gate, cand)
        bwd = affine_scan(gate.flip(1), cand.flip(1)).flip(1)
        combined = torch.tanh(enc + self.linear(torch.cat((fwd, bwd), dim=-1), "combine")) * v

        pooled = combined.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(emb.dtype)
        context = torch.sigmoid(self.linear(pooled, "global")) * pooled
        joined = torch.cat((combined, context.unsqueeze(1).expand_as(combined)), dim=-1)
        head_gate = torch.sigmoid(self.linear(joined, "head_gate"))
        hidden = torch.tanh(self.linear(torch.cat((joined, head_gate), dim=-1), "head_hidden"))
        out = self.linear(hidden, "output")
        return out[..., : self.roles], out[..., self.roles]
