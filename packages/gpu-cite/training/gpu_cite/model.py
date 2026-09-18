"""gpu-cite model: the shared ``ScanTagger`` family plus a CRF transition matrix.

    CiteTagger = ScanTagger(TABLE_ROWS, hidden=32, tags=31 + 3, pooled_out=8, scan_layers=2)

Everything the task needs fits the family through its escape hatches:

* the three extra tag columns carry the name-part head (``O`` / ``GIVEN`` / ``FAMILY``);
  ``split_tags`` separates them from the 31 BIO emissions,
* the pooled head carries the document type,
* a learned linear-chain CRF transition matrix ``trans`` [31, 31] is appended to the
  exported tensors and read only by the decoders (``evaluate.py``, ``src/decode.ts``);
  the family forward, ``scanTaggerForward`` and the canonical WGSL kernel never see it.

``padding_id`` is 0 because the featurizer pads every row with id 0 (row 0 of the table
is therefore never read), which keeps ``features.py``/``features.ts`` unchanged.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from gpu_utils_training.models import ScanTagger
from gpu_utils_training.qat import QParam
from torch import Tensor

from .features import TABLE_ROWS
from .labels import NAMEPARTS, TAGS, TYPES

HIDDEN = 32
PADDING_ID = 0
K = len(TAGS)
P = len(NAMEPARTS)


class CiteTagger(ScanTagger):
    """Scan family with name parts as extra tag columns, the type as the pooled head and a CRF."""

    def __init__(self, hidden: int = HIDDEN, head: int | None = None) -> None:
        super().__init__(TABLE_ROWS, hidden, K + P, pooled_out=len(TYPES), scan_layers=2, head=head, padding_id=PADDING_ID)
        self.trans = QParam(torch.zeros(K, K))

    def raw_tensors(self) -> OrderedDict[str, Tensor]:
        out = super().raw_tensors()
        out["trans"] = self.trans.w  # decoder-only extra tensor, exported after the family tensors
        return out

    def transitions(self) -> Tensor:
        """Learned CRF transitions ``[from, to]`` (fake-quantized when QAT is on)."""
        return self.trans()


def build(hidden: int = HIDDEN, head: int | None = None) -> CiteTagger:
    return CiteTagger(hidden, head)


def split_tags(logits: Tensor) -> tuple[Tensor, Tensor]:
    """``[..., K + P]`` tag logits -> (BIO emissions ``[..., K]``, name-part logits ``[..., P]``)."""
    return logits[..., :K], logits[..., K:]


def crf_nll(emissions: Tensor, trans: Tensor, gold: Tensor, mask: Tensor) -> Tensor:
    """Linear-chain CRF negative log-likelihood, mean over the batch.

    emissions [B, T, K], trans [K, K] (from -> to), gold [B, T] long (any valid id on
    padding), mask [B, T] bool or float prefix mask.
    """
    B, T, _ = emissions.shape
    m_all = mask.to(emissions.dtype)
    alpha = emissions[:, 0]  # [B, K]
    idx = torch.arange(B)
    score = emissions[:, 0].gather(1, gold[:, :1]).squeeze(1)
    for t in range(1, T):
        m = m_all[:, t].unsqueeze(-1)
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
        alpha = nxt * m + alpha * (1.0 - m)
        step = emissions[idx, t, gold[:, t]] + trans[gold[:, t - 1], gold[:, t]]
        score = score + step * m_all[:, t]
    return (torch.logsumexp(alpha, dim=1) - score).mean()
