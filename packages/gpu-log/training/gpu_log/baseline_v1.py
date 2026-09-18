"""The promoted v1 model, rebuilt from the artefacts committed on `main`.

v1 predates the shared families: it had its own module (a per-token kind head that was
mean-pooled afterwards, rather than the family's pooled head) and 23 labels, with no HOST
role. This module re-implements that forward pass over the int6 weights in
`main:packages/gpu-log/model/`, exposing the family interface (`padding_id`,
`forward(rows, mask) -> {"tags", "pooled"}`) so `evaluate.py` can score v1 and v2 with the
same code on the same sets. The featurizer is unchanged between v1 and v2, so the feature
rows are identical.

Nothing is committed: the v1 artefacts are read out of git on demand.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from gpu_utils_training.quant import decode_weights
from torch import Tensor, nn

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "v1"
REF = "main"
ARTEFACTS = ("manifest.json", "weights.txt")


def _artefact(name: str, ref: str = REF) -> str:
    """`git show <ref>:packages/gpu-log/model/<name>`, cached under data/cache/v1/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ref.replace('/', '_')}-{name}"
    if not path.exists():
        root = Path(__file__).resolve().parents[4]
        out = subprocess.run(
            ["git", "show", f"{ref}:packages/gpu-log/model/{name}"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        path.write_text(out, encoding="utf-8")
    return path.read_text(encoding="utf-8")


class V1Tagger(nn.Module):
    """v1's residual dilated CNN over dequantized int6 tensors (float32, eval only)."""

    def __init__(self, tensors: dict[str, torch.Tensor], blocks: int = 5) -> None:
        super().__init__()
        self.t = {k: v.float() for k, v in tensors.items()}
        self.n_blocks = blocks
        self.padding_id = int(self.t["embed"].shape[0])  # one past the table: contributes nothing

    def forward(self, rows: Tensor, mask: Tensor | None = None) -> dict[str, Tensor | None]:
        t = self.t
        keep = rows != self.padding_id
        emb = F.embedding(rows.masked_fill(~keep, 0), t["embed"]) * keep.unsqueeze(-1).float()
        x = emb.sum(2) @ t["proj_w"] + t["proj_b"]
        m = (mask.unsqueeze(1).float() if mask is not None else torch.ones(x.shape[0], 1, x.shape[1]))
        x = x.transpose(1, 2) * m  # [B, H, T]
        for i in range(self.n_blocks):
            d = 1 << i
            h = F.conv1d(x, t[f"block{i}_w1"].permute(2, 1, 0), t[f"block{i}_b1"], padding=d, dilation=d)
            y = F.conv1d(F.relu(h), t[f"block{i}_w2"].t().unsqueeze(-1), t[f"block{i}_b2"])
            x = (x + y) * m
        x = x.transpose(1, 2)  # [B, T, H]
        tags = F.relu(x @ t["head_h_w"] + t["head_h_b"]) @ t["head_tag_w"] + t["head_tag_b"]
        kind_tok = x @ t["head_kind_w"] + t["head_kind_b"]
        weights = m.transpose(1, 2)
        pooled = (kind_tok * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return {"tags": tags, "pooled": pooled}


def load_v1(ref: str = REF) -> tuple[V1Tagger, list[str]]:
    """(model, labels) for the promoted v1 checkpoint. Weights are the decoded int6 values."""
    manifest = json.loads(_artefact("manifest.json", ref))
    decoded = decode_weights(_artefact("weights.txt", ref), manifest)
    model = V1Tagger({k: torch.as_tensor(v) for k, v in decoded.items()}, blocks=manifest["blocks"])
    model.eval()
    return model, list(manifest["labels"])
