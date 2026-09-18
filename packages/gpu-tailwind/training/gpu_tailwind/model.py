"""Model definition: the shared scan family from gpu_utils_training.models.

ScanTagger(feature_rows, hidden, tags): summed sparse embeddings -> bidirectional gated
affine scans -> residual depthwise conv -> mean-pooled context -> two-layer head. The
tag head has one column per BIO role label plus one extra column for the segment-boundary
logit (split by the decoder), which is the family's sanctioned way to add a per-token head.
cpu.ts / gpu.ts use the runtime's family forward and canonical WGSL kernels unchanged.
"""

from __future__ import annotations

from gpu_utils_training.models import ScanTagger

from .features import ROWS
from .labels import LABELS

HIDDEN = 24
TAGS = len(LABELS) + 1  # roles + boundary column


def build() -> ScanTagger:
    return ScanTagger(ROWS, HIDDEN, TAGS)
