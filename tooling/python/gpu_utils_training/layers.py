"""Reference layers shared by every gpu-utils model family.

packages/runtime/src/layers.ts mirrors this file op for op; the shapes and conventions
below are the contract between the two.

Conventions
-----------
* Activations are ``[B, T, C]`` (batch, tokens, channels); ``mask`` is ``[B, T]`` bool
  with True for real tokens. Padded positions never influence real ones, so a batched
  forward equals the per-sequence forward the TypeScript runtime performs.
* Dense weights are stored ``[in, out]`` and applied as ``x @ W + b`` (row-major, so a
  GPU thread per output channel reads ``W[i * out + o]`` coalesced across threads).
* Gated scan: ``a = sigmoid(x Wa + ba)``, ``u = tanh(x Wu + bu)``,
  ``h_t = a_t * h_{t-1} + (1 - a_t) * u_t`` with ``h_0 = 0``. ``a`` is the *forget/keep*
  gate; ``(1 - a)`` scales the bounded candidate so the state stays in ``[-1, 1]`` and is
  robust to int6 quantization. This is the single convention used by every family.
* Depthwise convolution weights are ``[taps, C]``, zero padded, centre tap at
  ``taps // 2``; dilated block weights are ``[3, in, out]`` (tap, in, out).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .qat import QParam


def sparse_embed(rows: Tensor, table: Tensor, padding_id: int) -> Tensor:
    """Sum of embedding rows per token. rows [B, T, S] long, table [R, E] -> [B, T, E].

    Slots equal to ``padding_id`` contribute nothing (``padding_id`` may be ``R``, i.e. one
    past the table, which is the default for the families).
    """
    keep = rows != padding_id
    safe = rows.masked_fill(~keep, 0)
    emb = F.embedding(safe, table) * keep.unsqueeze(-1).to(table.dtype)
    return emb.sum(dim=2)


def affine_scan(a: Tensor, b: Tensor, mask: Tensor | None = None, reverse: bool = False) -> Tensor:
    """Inclusive scan ``h_t = a_t * h_{t-1} + b_t`` along dim 1 (Hillis-Steele doubling).

    a, b: ``[B, T, C]``. Padded positions (``mask`` False) become the identity map
    ``(a=1, b=0)`` so trailing padding never leaks into a reversed scan, and their output
    is zeroed. ``reverse=True`` scans from the last token towards the first.
    """
    if mask is not None:
        m = mask.unsqueeze(-1).to(a.dtype)
        a = a * m + (1.0 - m)
        b = b * m
    if reverse:
        a = a.flip(1)
        b = b.flip(1)
    n = a.shape[1]
    stride = 1
    while stride < n:
        a_prev = F.pad(a[:, :-stride], (0, 0, stride, 0), value=1.0)
        b_prev = F.pad(b[:, :-stride], (0, 0, stride, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        stride *= 2
    h = b.flip(1) if reverse else b
    if mask is not None:
        h = h * mask.unsqueeze(-1).to(h.dtype)
    return h


def _normal(shape: tuple[int, ...], std: float) -> Tensor:
    return torch.randn(shape) * std


class Dense(nn.Module):
    """``y = x @ W + b`` with W ``[in, out]``."""

    def __init__(self, d_in: int, d_out: int, std: float | None = None) -> None:
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.w = QParam(_normal((d_in, d_out), std if std is not None else 1.0 / math.sqrt(d_in)))
        self.b = QParam(torch.zeros(d_out))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w() + self.b()


class BiScan(nn.Module):
    """Gated bidirectional affine scan: ``[B, T, d_in] -> [B, T, 2 * hidden]`` (forward ‖ backward).

    Per direction: ``a = sigmoid(x Wa + ba)``, ``u = tanh(x Wu + bu)``,
    ``h_t = a_t * h_{t-1} + (1 - a_t) * u_t``. Gate biases are initialised on a ramp so the
    channels start with a spread of memory lengths (mined from gpu-view).
    """

    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.d_in, self.hidden = d_in, hidden
        std = 1.0 / math.sqrt(d_in)
        for direction in ("f", "b"):
            setattr(self, f"{direction}_wa", QParam(_normal((d_in, hidden), std)))
            setattr(self, f"{direction}_ba", QParam(torch.linspace(0.0, 3.0, hidden)))
            setattr(self, f"{direction}_wu", QParam(_normal((d_in, hidden), std)))
            setattr(self, f"{direction}_bu", QParam(torch.zeros(hidden)))

    def direction(self, x: Tensor, mask: Tensor | None, reverse: bool) -> Tensor:
        p = "b" if reverse else "f"
        a = torch.sigmoid(x @ getattr(self, f"{p}_wa")() + getattr(self, f"{p}_ba")())
        u = torch.tanh(x @ getattr(self, f"{p}_wu")() + getattr(self, f"{p}_bu")())
        return affine_scan(a, (1.0 - a) * u, mask, reverse)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return torch.cat([self.direction(x, mask, False), self.direction(x, mask, True)], dim=-1)

    def tensors(self, prefix: str) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for d in ("f", "b"):
            for n in ("wa", "ba", "wu", "bu"):
                out[f"{prefix}.{d}.{n}"] = getattr(self, f"{d}_{n}").w
        return out


class DepthwiseConv(nn.Module):
    """Per-channel 1-D convolution over time with ``taps`` (default 5) zero-padded taps.

    ``w[k, c]`` multiplies ``x[t + k - taps // 2, c]``. Padded/out-of-range tokens read as 0.
    """

    def __init__(self, channels: int, taps: int = 5, std: float = 0.1) -> None:
        super().__init__()
        self.channels, self.taps = channels, taps
        self.w = QParam(_normal((taps, channels), std))
        self.b = QParam(torch.zeros(channels))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        half = self.taps // 2
        w = self.w()
        xp = F.pad(x, (0, 0, half, self.taps - 1 - half))
        y = self.b().expand_as(x)
        for k in range(self.taps):
            y = y + xp[:, k : k + x.shape[1]] * w[k]
        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)
        return y


class DilatedResidualBlock(nn.Module):
    """``y = x + relu(conv3_dilated(x)) @ W2 + b2`` with zero padding; output masked.

    ``w1`` is ``[3, hidden, hidden]`` (tap, in, out): tap 0 reads ``t - d``, tap 2 ``t + d``.
    """

    def __init__(self, hidden: int, dilation: int) -> None:
        super().__init__()
        self.hidden, self.dilation = hidden, dilation
        self.w1 = QParam(_normal((3, hidden, hidden), math.sqrt(1.0 / (3 * hidden))))
        self.b1 = QParam(torch.zeros(hidden))
        self.w2 = QParam(_normal((hidden, hidden), math.sqrt(0.5 / hidden)))
        self.b2 = QParam(torch.zeros(hidden))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        m = None if mask is None else mask.unsqueeze(-1).to(x.dtype)
        if m is not None:
            x = x * m
        d = self.dilation
        w1 = self.w1()
        xp = F.pad(x, (0, 0, d, d))
        n = x.shape[1]
        h = self.b1() + xp[:, :n] @ w1[0] + x @ w1[1] + xp[:, 2 * d :] @ w1[2]
        y = x + torch.relu(h) @ self.w2() + self.b2()
        return y if m is None else y * m

    def tensors(self, prefix: str) -> dict[str, Tensor]:
        return {
            f"{prefix}.w1": self.w1.w,
            f"{prefix}.b1": self.b1.w,
            f"{prefix}.w2": self.w2.w,
            f"{prefix}.b2": self.b2.w,
        }


def masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over the token axis of the real tokens: ``[B, T, C] -> [B, C]``."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
