"""Quantization-aware training helpers for int6 (or int8) export.

`fake_quant` is bit-identical to what the runtime decodes: the scale is computed in
float64 exactly like quant.quantize, the codes are rounded half-to-even like NumPy, and
the dequantized value is float32(code * scale64) like runtime/weights.ts. That means a
model evaluated with QAT enabled reproduces the exported weights *exactly*, so fixtures
and parity tests can be generated from the torch model directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn


def levels(bits: int) -> int:
    return (1 << (bits - 1)) - 1


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, w: Tensor, bits: int) -> Tensor:  # type: ignore[override]
        lv = levels(bits)
        scale64 = w.detach().abs().max().double() / lv
        if scale64 == 0:
            scale64 = torch.tensor(1.0, dtype=torch.float64)
        scale32 = scale64.to(torch.float32)
        q = torch.clamp(torch.round(w / scale32), -lv, lv)
        return (q.double() * scale64).to(torch.float32)

    @staticmethod
    def backward(ctx: Any, g: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return g, None  # straight-through estimator


def fake_quant(w: Tensor, bits: int = 6) -> Tensor:
    """Symmetric per-tensor fake quantization with a straight-through gradient."""
    return _FakeQuant.apply(w, bits)  # type: ignore[no-any-return]


class QParam(nn.Module):
    """A learnable tensor that is fake-quantized on read once `quant` is switched on.

    Call the module to read the (possibly quantized) value: `w = self.embed()`.
    """

    def __init__(self, value: Tensor, bits: int = 6) -> None:
        super().__init__()
        self.w = nn.Parameter(value)
        self.bits = bits
        self.quant = False

    def forward(self) -> Tensor:
        return fake_quant(self.w, self.bits) if self.quant else self.w

    @property
    def shape(self) -> torch.Size:
        return self.w.shape

    def extra_repr(self) -> str:
        return f"shape={tuple(self.w.shape)}, bits={self.bits}, quant={self.quant}"


def qparams(model: nn.Module) -> Iterator[tuple[str, QParam]]:
    for name, m in model.named_modules():
        if isinstance(m, QParam):
            yield name, m


def set_quant(model: nn.Module, on: bool, bits: int | None = None) -> None:
    """Toggle fake quantization on every QParam in the model (optionally changing the width)."""
    for _, q in qparams(model):
        q.quant = on
        if bits is not None:
            q.bits = bits


class QuantMixin:
    """Mixin for nn.Modules built from QParams: `model.quant = True` toggles every one."""

    @property
    def quant(self) -> bool:
        return all(q.quant for _, q in qparams(self))  # type: ignore[arg-type]

    @quant.setter
    def quant(self, on: bool) -> None:
        set_quant(self, on)  # type: ignore[arg-type]

    def parameter_count(self) -> int:
        return sum(q.w.numel() for _, q in qparams(self))  # type: ignore[arg-type]
