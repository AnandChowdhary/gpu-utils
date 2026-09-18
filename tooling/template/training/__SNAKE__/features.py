"""Tokenizer and sparse featurizer. Must match src/features.ts exactly.

Each token gets SLOTS feature ids into one embedding table of FEATURE_ROWS rows; the
blocks below give every feature family its own id range so they never collide.
"""

from __future__ import annotations

from gpu_utils_training.features import Token, hash_token, tokenize  # shared reference impl

HASH_BUCKETS = 1024
BLOCKS: list[tuple[str, int]] = [("word", HASH_BUCKETS), ("shape", 8), ("length", 16)]
OFFSET: dict[str, int] = {}
_o = 0
for _name, _size in BLOCKS:
    OFFSET[_name] = _o
    _o += _size
FEATURE_ROWS = _o
SLOTS = len(BLOCKS)
LABELS = ["O", "B-NAME", "I-NAME"]


def featurize_tokens(tokens: list[Token]) -> list[list[int]]:
    return [
        [
            OFFSET["word"] + hash_token(t.text, HASH_BUCKETS),
            OFFSET["shape"] + t.shape,
            OFFSET["length"] + min(len(t.text), 15),
        ]
        for t in tokens
    ]


def featurize(text: str) -> list[list[int]]:
    return featurize_tokens(tokenize(text))
