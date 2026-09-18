"""Model definition: a shared family from gpu_utils_training.models.

ScanTagger (bidirectional gated affine scans) for short natural-language inputs, or
ConvTagger (residual dilated 1-D convolutions) for long documents. Both export tensors in
the layout packages/runtime/src/models.ts and the canonical WGSL kernels read, so cpu.ts
and gpu.ts need no model-specific code. Only write a custom nn.Module when the README
explains why the families do not fit.
"""

from __future__ import annotations

from gpu_utils_training.models import ScanTagger

from .features import FEATURE_ROWS, LABELS

HIDDEN = 32


def build() -> ScanTagger:
    return ScanTagger(FEATURE_ROWS, HIDDEN, len(LABELS))
