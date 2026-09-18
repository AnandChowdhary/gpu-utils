"""gpu-paste model: the shared scan family from gpu_utils_training.models.

    ScanTagger(TOTAL_ROWS, HIDDEN, len(LABELS), pooled_out=len(LEARNED_KINDS))

Summed sparse embeddings → one bidirectional gated affine scan layer → residual 5-tap
depthwise conv → mean-pooled context → two-layer per-token head and two-layer pooled head.
The per-token ``tags`` output is the BIO span head (O + B/I × six span kinds) and the
``pooled`` output is the six-way kind head; src/decode.ts and src/index.ts split them.
The CPU path (runtime ``scanTaggerForward``) and the WGSL path (runtime ``runScanTagger``)
are the runtime's canonical implementations, so this file only picks the sizes.
"""

from __future__ import annotations

from gpu_utils_training.models import ScanTagger

from gpu_paste.data import LABELS, LEARNED_KINDS
from gpu_paste.features import TOTAL_ROWS

HIDDEN = 32


def build() -> ScanTagger:
    return ScanTagger(TOTAL_ROWS, HIDDEN, len(LABELS), pooled_out=len(LEARNED_KINDS))
