"""Residual dilated-CNN token tagger with a line-kind head, int6 quantization-aware.

Layout mirrors src/cpu.ts: summed sparse embeddings (E) -> linear to H -> 5 residual
blocks [conv3(dilation d) -> relu -> conv1 -> +x] -> tag head (H -> H -> tags) and a
per-token kind head (H -> kinds) whose logits are mean-pooled over the line.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import EMBED_ROWS, FEATURE_COUNT

LEVELS = 31


def fake_quant(w: torch.Tensor) -> torch.Tensor:
    """int6 symmetric per-tensor round-trip with a straight-through estimator."""
    scale = w.detach().abs().max().clamp_min(1e-8) / LEVELS
    q = torch.clamp(torch.round(w / scale), -LEVELS, LEVELS) * scale
    return w + (q - w).detach()


class QParam(nn.Module):
    def __init__(self, shape: tuple[int, ...], std: float | None = None, zero: bool = False):
        super().__init__()
        if zero:
            self.w = nn.Parameter(torch.zeros(shape))
        else:
            fan_in = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]
            self.w = nn.Parameter(torch.randn(shape) * (std if std is not None else (2.0 / fan_in) ** 0.5))
        self.quant = False

    def forward(self) -> torch.Tensor:
        return fake_quant(self.w) if self.quant else self.w


class Block(nn.Module):
    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.w1 = QParam((3, hidden, hidden), std=(1.0 / (3 * hidden)) ** 0.5)  # [tap, in, out]
        self.b1 = QParam((hidden,), zero=True)
        self.w2 = QParam((hidden, hidden), std=(0.5 / hidden) ** 0.5)  # [in, out]
        self.b2 = QParam((hidden,), zero=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L], mask: [B, 1, L]
        w1 = self.w1().permute(2, 1, 0)  # [out, in, tap]
        h = F.conv1d(x, w1, self.b1(), padding=self.dilation, dilation=self.dilation)
        h = F.relu(h)
        y = F.conv1d(h, self.w2().t().unsqueeze(-1), self.b2())
        return (x + y) * mask


class LogTagger(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 64, blocks: int = 5, n_tags: int = 23, n_kinds: int = 3):
        super().__init__()
        self.embed_dim, self.hidden, self.n_tags, self.n_kinds = embed_dim, hidden, n_tags, n_kinds
        self.embed = QParam((EMBED_ROWS, embed_dim), std=0.3)
        self.proj_w = QParam((embed_dim, hidden))
        self.proj_b = QParam((hidden,), zero=True)
        self.blocks = nn.ModuleList([Block(hidden, 2**i) for i in range(blocks)])
        self.head_h_w = QParam((hidden, hidden))
        self.head_h_b = QParam((hidden,), zero=True)
        self.head_tag_w = QParam((hidden, n_tags))
        self.head_tag_b = QParam((n_tags,), zero=True)
        self.head_kind_w = QParam((hidden, n_kinds))
        self.head_kind_b = QParam((n_kinds,), zero=True)

    def set_quant(self, on: bool) -> None:
        for m in self.modules():
            if isinstance(m, QParam):
                m.quant = on

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: [B, L, F] int64, mask: [B, L] float. Returns tag logits [B, L, T], kind logits [B, K]."""
        x = F.embedding(feats, self.embed()).sum(2)  # [B, L, E]
        x = x @ self.proj_w() + self.proj_b()
        m = mask.unsqueeze(1)
        x = (x.transpose(1, 2)) * m  # [B, H, L]
        for blk in self.blocks:
            x = blk(x, m)
        x = x.transpose(1, 2)  # [B, L, H]
        h = F.relu(x @ self.head_h_w() + self.head_h_b())
        tags = h @ self.head_tag_w() + self.head_tag_b()
        kind_tok = x @ self.head_kind_w() + self.head_kind_b()  # [B, L, K]
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        kind = (kind_tok * mask.unsqueeze(-1)).sum(1) / denom
        return tags, kind

    def tensors(self) -> dict[str, np.ndarray]:
        """Export layout (names and shapes are what src/model.ts expects)."""
        out: dict[str, np.ndarray] = {
            "embed": self.embed.w.detach().numpy(),
            "proj_w": self.proj_w.w.detach().numpy(),
            "proj_b": self.proj_b.w.detach().numpy(),
        }
        for i, blk in enumerate(self.blocks):
            out[f"block{i}_w1"] = blk.w1.w.detach().numpy()
            out[f"block{i}_b1"] = blk.b1.w.detach().numpy()
            out[f"block{i}_w2"] = blk.w2.w.detach().numpy()
            out[f"block{i}_b2"] = blk.b2.w.detach().numpy()
        for name in ("head_h_w", "head_h_b", "head_tag_w", "head_tag_b", "head_kind_w", "head_kind_b"):
            out[name] = getattr(self, name).w.detach().numpy()
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def forward_numpy(t: dict[str, np.ndarray], feats: np.ndarray, blocks: int = 5) -> np.ndarray:
    """Reference float32 forward for ONE line from exported (dequantized) tensors.

    feats: [L, F]. Returns [L, T + K]: tag logits then per-token kind logits. Accumulation
    order is exactly the one src/cpu.ts uses (sequential over inputs, float32 multiply then
    add), so model/fixtures.json reproduces the TypeScript path to ~1e-6 rather than BLAS
    reordering noise.
    """
    feats = np.asarray(feats)
    L = feats.shape[0]
    f32 = np.float32
    embed = t["embed"].astype(f32)
    E = embed.shape[1]
    x = np.zeros((L, t["proj_w"].shape[1]), dtype=f32)
    for p in range(L):
        s = np.zeros(E, dtype=f32)
        for f in feats[p]:
            s += embed[f]
        acc = t["proj_b"].astype(f32).copy()
        for e in range(E):
            acc += f32(s[e]) * t["proj_w"][e].astype(f32)
        x[p] = acc
    H = x.shape[1]
    for i in range(blocks):
        d = 1 << i
        w1, b1, w2, b2 = (t[f"block{i}_w1"].astype(f32), t[f"block{i}_b1"].astype(f32), t[f"block{i}_w2"].astype(f32), t[f"block{i}_b2"].astype(f32))
        y = np.zeros_like(x)
        for p in range(L):
            h = b1.copy()
            for tap, off in enumerate((-d, 0, d)):
                q = p + off
                if 0 <= q < L:
                    for j in range(H):
                        v = x[q, j]
                        if v != 0:
                            h += v * w1[tap, j]
            h = np.maximum(h, f32(0))
            acc = x[p] + b2
            for j in range(H):
                if h[j] > 0:
                    acc += h[j] * w2[j]
            y[p] = acc
        x = y
    T = t["head_tag_w"].shape[1]
    K = t["head_kind_w"].shape[1]
    out = np.zeros((L, T + K), dtype=f32)
    hw, hb = t["head_h_w"].astype(f32), t["head_h_b"].astype(f32)
    tw, tb = t["head_tag_w"].astype(f32), t["head_tag_b"].astype(f32)
    kw, kb = t["head_kind_w"].astype(f32), t["head_kind_b"].astype(f32)
    for p in range(L):
        hh = hb.copy()
        for j in range(H):
            hh += x[p, j] * hw[j]
        hh = np.maximum(hh, f32(0))
        tags = tb.copy()
        for j in range(H):
            if hh[j] > 0:
                tags += hh[j] * tw[j]
        kind = kb.copy()
        for j in range(H):
            kind += x[p, j] * kw[j]
        out[p, :T] = tags
        out[p, T:] = kind
    return out


def dequantized(t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from gpu_utils_training.quant import quantize

    out = {}
    for k, v in t.items():
        q, scale = quantize(np.asarray(v, dtype=np.float32))
        out[k] = (q.astype(np.float32) * np.float32(scale)).astype(np.float32)
    return out


__all__ = ["FEATURE_COUNT", "LogTagger", "dequantized", "fake_quant", "forward_numpy"]
