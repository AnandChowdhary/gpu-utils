"""Model definition: the shared scan family (gpu_utils_training.models.ScanTagger).

Tags are the 14 token roles plus one extra column carrying the clause-boundary logit, so
the family's per-token head serves both outputs; the decoder splits the columns.
"""

from __future__ import annotations

from gpu_utils_training.models import ScanTagger

from .features import FEATURE_ROWS
from .generate import ROLES

HIDDEN = 32
TAGS = len(ROLES) + 1  # roles + clause-boundary logit


def build() -> ScanTagger:
    return ScanTagger(FEATURE_ROWS, HIDDEN, TAGS)
