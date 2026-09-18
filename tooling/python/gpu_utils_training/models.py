"""The two reference model families. Both share one interface:

    out = model(rows, mask)          # rows [B, T, slots] long, mask [B, T] bool
    out["tags"]                      # [B, T, tags] per-token logits
    out["pooled"]                    # [B, pooled_out] sequence logits, or None

``tensors()`` returns the exported tensors in the exact order/layout that
packages/runtime/src/models.ts and the canonical WGSL kernels read them; ``config()`` is
what export_package copies into manifest.json. Subclass a family (or wrap one) when a
package needs an extra head, and say why in the package README.

Scan family (short natural-language inputs, ~30-60K params)::

    e      = sparse_embed(rows)                              [T, H]
    for each scan layer l:
        h  = BiScan_l(x)                                     [T, 2H]   (x = e for l = 0)
        x  = h + relu(DepthwiseConv5_l(h))                   [T, 2H]   residual local mixing
    ctx    = mean_t x                                        [2H]
    g      = relu([x_t ‖ ctx] W_head + b_head)               [T, head]  (head = 2H)
    tags   = g W_tags + b_tags                               [T, tags]
    pooled = relu(ctx W_p1 + b_p1) W_p2 + b_p2               [pooled_out]

Conv family (long documents, 100K-1M params)::

    e      = sparse_embed(rows)                              [T, E]
    x      = e W_proj + b_proj                               [T, H]
    for each block i (dilation d_i):
        x  = x + relu(conv3_{d_i}(x)) W2 + b2                [T, H]
    g      = relu(x W_head + b_head)                         [T, head]  (head = H)
    tags   = g W_tags + b_tags                               [T, tags]
    ctx    = mean_t x ;  pooled = relu(ctx W_p1 + b_p1) W_p2 + b_p2
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .layers import (
    BiScan,
    Dense,
    DepthwiseConv,
    DilatedResidualBlock,
    masked_mean,
    sparse_embed,
)
from .qat import QParam, QuantMixin

Output = dict[str, Tensor | None]


class TaggerBase(QuantMixin, nn.Module):
    """Shared plumbing: pooled head, tensor export/import, config."""

    family = ""

    def __init__(self, feature_rows: int, embed: int, tags: int, pooled_out: int, padding_id: int | None, embed_std: float) -> None:
        super().__init__()
        self.feature_rows = feature_rows
        self.embed_dim = embed
        self.n_tags = tags
        self.pooled_out = pooled_out
        self.padding_id = feature_rows if padding_id is None else padding_id
        self.embed = QParam(torch.randn(feature_rows, embed) * embed_std)

    def _init_heads(self, width: int, head: int) -> None:
        self.head_dim = head
        self.head = Dense(width, head)
        self.tags = Dense(head, self.n_tags)
        if self.pooled_out:
            self.pool1 = Dense(self.pool_in, head)
            self.pool2 = Dense(head, self.pooled_out)

    def _pooled(self, ctx: Tensor) -> Tensor | None:
        if not self.pooled_out:
            return None
        return self.pool2(torch.relu(self.pool1(ctx)))

    def _head_tensors(self) -> dict[str, Tensor]:
        out = {"head.w": self.head.w.w, "head.b": self.head.b.w, "tags.w": self.tags.w.w, "tags.b": self.tags.b.w}
        if self.pooled_out:
            out.update({"pool.w1": self.pool1.w.w, "pool.b1": self.pool1.b.w, "pool.w2": self.pool2.w.w, "pool.b2": self.pool2.b.w})
        return out

    def raw_tensors(self) -> OrderedDict[str, Tensor]:
        raise NotImplementedError

    def tensors(self) -> OrderedDict[str, np.ndarray]:
        """Ordered name -> float32 array, in the layout the TS/WGSL runtimes read."""
        return OrderedDict((k, v.detach().cpu().numpy().astype(np.float32)) for k, v in self.raw_tensors().items())

    def tensor_names(self) -> list[str]:
        return list(self.raw_tensors().keys())

    @torch.no_grad()
    def load_tensors(self, tensors: dict[str, np.ndarray]) -> None:
        """Copy exported (e.g. decoded int6) tensors back into the parameters."""
        for name, param in self.raw_tensors().items():
            param.copy_(torch.as_tensor(np.asarray(tensors[name], dtype=np.float32)).reshape(param.shape))

    def config(self) -> dict[str, Any]:
        raise NotImplementedError

    def forward(self, rows: Tensor, mask: Tensor | None = None) -> Output:
        raise NotImplementedError


class ScanTagger(TaggerBase):
    family = "scan"

    def __init__(
        self,
        feature_rows: int,
        hidden: int,
        tags: int,
        pooled_out: int = 0,
        scan_layers: int = 1,
        *,
        head: int | None = None,
        padding_id: int | None = None,
        conv_taps: int = 5,
        embed_std: float = 0.1,
    ) -> None:
        super().__init__(feature_rows, hidden, tags, pooled_out, padding_id, embed_std)
        self.hidden = hidden
        self.scan_layers = scan_layers
        self.conv_taps = conv_taps
        width = 2 * hidden
        self.scans = nn.ModuleList([BiScan(hidden if i == 0 else width, hidden) for i in range(scan_layers)])
        self.convs = nn.ModuleList([DepthwiseConv(width, conv_taps) for _ in range(scan_layers)])
        self.pool_in = width
        self._init_heads(2 * width, head if head is not None else width)

    def forward(self, rows: Tensor, mask: Tensor | None = None) -> Output:
        x = sparse_embed(rows, self.embed(), self.padding_id)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        for scan, conv in zip(self.scans, self.convs, strict=True):
            h = scan(x, mask)
            x = h + torch.relu(conv(h, mask))
        ctx = masked_mean(x, mask)
        joined = torch.cat([x, ctx.unsqueeze(1).expand_as(x)], dim=-1)
        g = torch.relu(self.head(joined))
        return {"tags": self.tags(g), "pooled": self._pooled(ctx)}

    def raw_tensors(self) -> OrderedDict[str, Tensor]:
        out: OrderedDict[str, Tensor] = OrderedDict(embed=self.embed.w)
        for i, (scan, conv) in enumerate(zip(self.scans, self.convs, strict=True)):
            out.update(scan.tensors(f"scan{i}"))
            out[f"conv{i}.w"] = conv.w.w
            out[f"conv{i}.b"] = conv.b.w
        out.update(self._head_tensors())
        return out

    def config(self) -> dict[str, Any]:
        return {
            "family": "scan",
            "featureRows": self.feature_rows,
            "paddingId": self.padding_id,
            "hidden": self.hidden,
            "head": self.head_dim,
            "scanLayers": self.scan_layers,
            "convTaps": self.conv_taps,
            "tags": self.n_tags,
            "pooled": self.pooled_out,
        }


class ConvTagger(TaggerBase):
    family = "conv"

    def __init__(
        self,
        feature_rows: int,
        embed: int,
        hidden: int,
        blocks: int,
        dilations: list[int] | None,
        tags: int,
        pooled_out: int = 0,
        *,
        head: int | None = None,
        padding_id: int | None = None,
        embed_std: float = 0.3,
    ) -> None:
        super().__init__(feature_rows, embed, tags, pooled_out, padding_id, embed_std)
        self.hidden = hidden
        self.dilations = list(dilations) if dilations is not None else [2**i for i in range(blocks)]
        if len(self.dilations) != blocks:
            raise ValueError(f"blocks={blocks} but {len(self.dilations)} dilations given")
        self.proj = Dense(embed, hidden)
        self.blocks = nn.ModuleList([DilatedResidualBlock(hidden, d) for d in self.dilations])
        self.pool_in = hidden
        self._init_heads(hidden, head if head is not None else hidden)

    def forward(self, rows: Tensor, mask: Tensor | None = None) -> Output:
        x = self.proj(sparse_embed(rows, self.embed(), self.padding_id))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        g = torch.relu(self.head(x))
        return {"tags": self.tags(g), "pooled": self._pooled(masked_mean(x, mask))}

    def raw_tensors(self) -> OrderedDict[str, Tensor]:
        out: OrderedDict[str, Tensor] = OrderedDict(embed=self.embed.w)
        out["proj.w"] = self.proj.w.w
        out["proj.b"] = self.proj.b.w
        for i, block in enumerate(self.blocks):
            out.update(block.tensors(f"block{i}"))
        out.update(self._head_tensors())
        return out

    def config(self) -> dict[str, Any]:
        return {
            "family": "conv",
            "featureRows": self.feature_rows,
            "paddingId": self.padding_id,
            "embed": self.embed_dim,
            "hidden": self.hidden,
            "head": self.head_dim,
            "dilations": self.dilations,
            "tags": self.n_tags,
            "pooled": self.pooled_out,
        }


def from_config(config: dict[str, Any]) -> TaggerBase:
    """Rebuild a family model from a manifest/config dict (e.g. to load decoded weights)."""
    common = {"padding_id": config["paddingId"], "head": config["head"]}
    if config["family"] == "scan":
        return ScanTagger(config["featureRows"], config["hidden"], config["tags"], config["pooled"], config["scanLayers"], conv_taps=config["convTaps"], **common)
    if config["family"] == "conv":
        d = list(config["dilations"])
        return ConvTagger(config["featureRows"], config["embed"], config["hidden"], len(d), d, config["tags"], config["pooled"], **common)
    raise ValueError(f"unknown family {config['family']!r}")
