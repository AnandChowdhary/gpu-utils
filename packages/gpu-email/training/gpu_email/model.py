"""Model definition: the shared conv family from gpu_utils_training.models.

gpu-email is a whole-document tagger, so it uses ``ConvTagger`` (summed sparse
embeddings -> projection -> six residual dilated 1-D convolution blocks with dilations
1..32, a 127-token receptive field -> per-token head). The two task heads are tag
columns of one output: the first ``len(LINE_KINDS)`` logits are the line-kind classifier
(decided per line by averaging token log-probabilities in the decoder) and the remaining
``len(BIO_LABELS)`` are the BIO contact-field tagger. The runtime's ``convTaggerForward``
and the canonical ``conv_tagger.wgsl`` kernel run it unchanged; the decoder splits the
logits (see decode.py / src/decode.ts).
"""

from __future__ import annotations

from gpu_utils_training.models import ConvTagger

from gpu_email.features import BIO_LABELS, LINE_KINDS, NUM_ROWS

EMBED = 48
HIDDEN = 48
DILATIONS = [1, 2, 4, 8, 16, 32]
N_KINDS = len(LINE_KINDS)
N_BIO = len(BIO_LABELS)


def build() -> ConvTagger:
    return ConvTagger(
        NUM_ROWS, EMBED, HIDDEN, len(DILATIONS), DILATIONS, N_KINDS + N_BIO
    )
