import numpy as np
import torch

from gpu_utils_training import quant
from gpu_utils_training.qat import QParam, QuantMixin, fake_quant, set_quant


def test_fake_quant_matches_runtime_decoding() -> None:
    torch.manual_seed(0)
    for shape in [(7,), (5, 9), (3, 4, 6)]:
        w = torch.randn(shape) * 0.37
        expected = quant.fake_quant(w.numpy())
        assert np.array_equal(fake_quant(w).numpy(), expected)


def test_fake_quant_is_straight_through() -> None:
    w = torch.randn(16, requires_grad=True)
    fake_quant(w).sum().backward()
    assert torch.equal(w.grad, torch.ones(16))


def test_fake_quant_zero_tensor_and_bits() -> None:
    assert torch.equal(fake_quant(torch.zeros(4)), torch.zeros(4))
    w = torch.linspace(-1, 1, 9)
    assert len(set(fake_quant(w, bits=8).tolist())) >= len(set(fake_quant(w, bits=4).tolist()))


class Tiny(QuantMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = QParam(torch.randn(4, 4))
        self.b = QParam(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a() + self.b()


def test_qparam_toggle() -> None:
    m = Tiny()
    x = torch.randn(2, 4)
    assert not m.quant
    plain = m(x)
    set_quant(m, True)
    assert m.quant
    assert not torch.equal(plain, m(x))
    assert torch.equal(m(x), x @ fake_quant(m.a.w) + fake_quant(m.b.w))
    m.quant = False
    assert torch.equal(m(x), plain)
    assert m.parameter_count() == 20
