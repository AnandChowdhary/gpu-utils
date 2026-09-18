"""Model definition: the shared conv family (gpu_utils_training.models.ConvTagger).

Summed sparse embeddings (32) → projection to 64 → five residual dilated conv blocks
(dilations 1, 2, 4, 8, 16) → per-token tag head (25 BIO labels) and a mean-pooled head
for the line kind (entry / continuation / frame). Tensors export in the layout that
packages/runtime/src/models.ts and wgsl/conv_tagger.wgsl read, so cpu.ts and gpu.ts are the
runtime's reference implementations.
"""

from __future__ import annotations

from gpu_utils_training.models import ConvTagger

from .data import KINDS, LABELS
from .features import EMBED_ROWS

EMBED = 32
HIDDEN = 64
DILATIONS = [1, 2, 4, 8, 16]


def build() -> ConvTagger:
    return ConvTagger(EMBED_ROWS, EMBED, HIDDEN, len(DILATIONS), DILATIONS, len(LABELS), pooled_out=len(KINDS))
