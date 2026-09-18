import numpy as np
import torch

from gpu_view import features
from gpu_view.export import forward_numpy
from gpu_view.generate import ROLES
from gpu_view.model import ViewTagger, affine_scan


def test_affine_scan_matches_sequential() -> None:
    torch.manual_seed(0)
    gate = torch.rand(2, 9, 4)
    cand = torch.randn(2, 9, 4)
    out = affine_scan(gate, cand)
    state = torch.zeros(2, 4)
    for t in range(9):
        state = gate[:, t] * state + cand[:, t]
        assert torch.allclose(out[:, t], state, atol=1e-6)


def test_numpy_reference_matches_torch() -> None:
    torch.manual_seed(1)
    model = ViewTagger(features.FEATURE_ROWS, len(ROLES)).double()
    schema = {"fields": [{"name": "status", "kind": "enum", "values": ["open"]}, {"name": "age", "kind": "number"}]}
    _toks, rows = features.featurize("open items with age over 30 sorted by age", schema)
    w = {name: getattr(model, name).detach().numpy() for name in ViewTagger.TENSOR_NAMES}
    ref = forward_numpy(w, rows)
    padded = np.full((1, len(rows), features.SLOTS), features.PADDING_ROW, dtype=np.int64)
    for i, r in enumerate(rows):
        padded[0, i, : len(r)] = r
    with torch.no_grad():
        tr, tb = model(torch.from_numpy(padded), torch.ones(1, len(rows), dtype=torch.bool))
    got = torch.cat([tr[0], tb[0].unsqueeze(-1)], dim=-1).numpy()
    assert np.abs(got - ref).max() < 1e-9
    assert model.parameter_count() < 40000
