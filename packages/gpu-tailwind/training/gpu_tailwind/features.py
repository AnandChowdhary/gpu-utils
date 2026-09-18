"""Tokenizer and sparse featurizer. Must match src/features.ts exactly.

Each token -> 7 feature ids into one embedding table of ROWS rows:
  word hash (1024) | consonant-skeleton hash (256) | 3-char prefix hash (128) |
  3-char suffix hash (128) | shape (8) | length bucket (16) | char class (5)
"""

from __future__ import annotations

import re

from gpu_utils_training.features import Token, hash_token, tokenize

WORD, SKEL, PRE, SUF = 1024, 256, 128, 128
OFF_SKEL = WORD
OFF_PRE = OFF_SKEL + SKEL
OFF_SUF = OFF_PRE + PRE
OFF_SHAPE = OFF_SUF + SUF
OFF_LEN = OFF_SHAPE + 8
OFF_CLS = OFF_LEN + 16
ROWS = OFF_CLS + 5
WIDTH = 7


def skeleton(text: str) -> str:
    t = text.lower()
    if not t:
        return ""
    body = re.sub(r"[aeiou]", "", t[1:])
    s = t[0] + body
    return re.sub(r"(.)\1+", r"\1", s)


def featurize_tokens(tokens: list[Token]) -> list[list[int]]:
    rows: list[list[int]] = []
    for t in tokens:
        txt = t.text
        rows.append(
            [
                hash_token(txt, WORD),
                OFF_SKEL + hash_token(skeleton(txt), SKEL),
                OFF_PRE + hash_token(txt[:3], PRE),
                OFF_SUF + hash_token(txt[-3:], SUF),
                OFF_SHAPE + t.shape,
                OFF_LEN + min(len(txt), 15),
                OFF_CLS + t.cls,
            ]
        )
    return rows


def featurize(text: str) -> list[list[int]]:
    return featurize_tokens(tokenize(text))
