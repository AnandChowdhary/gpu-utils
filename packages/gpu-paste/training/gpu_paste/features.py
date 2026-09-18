"""Tokenizer and sparse featurizer. Must match src/features.ts exactly.

Every token gets FEATURE_COUNT hashed/bucketed ids that index one flat embedding table
(TOTAL_ROWS rows). The ids are:

  0  word      hash of the token text                          (1024 buckets)
  1  skeleton  hash of the consonant skeleton                  (256 buckets)
  2  shape     tokenizer shape class                           (8)
  3  prefix    hash of the first two code points               (128)
  4  suffix    hash of the last two code points                (128)
  5  length    UTF-16 length, capped                           (16)
  6  column    token index within its line, capped             (6)
  7  line      line index from the start of the text, capped   (5)
  8  fromEnd   line index from the end of the text, capped     (3)
  9  bias      constant                                        (1)
"""

from __future__ import annotations

from gpu_utils_training.features import CLASS_LETTER, CLASS_NEWLINE, Token, hash_token, tokenize

WORD_BUCKETS = 1024
SKELETON_BUCKETS = 256
SHAPE_ROWS = 8
AFFIX_BUCKETS = 128
LENGTH_ROWS = 16
COLUMN_ROWS = 6
LINE_ROWS = 5
FROM_END_ROWS = 3

BASE_WORD = 0
BASE_SKELETON = BASE_WORD + WORD_BUCKETS
BASE_SHAPE = BASE_SKELETON + SKELETON_BUCKETS
BASE_PREFIX = BASE_SHAPE + SHAPE_ROWS
BASE_SUFFIX = BASE_PREFIX + AFFIX_BUCKETS
BASE_LENGTH = BASE_SUFFIX + AFFIX_BUCKETS
BASE_COLUMN = BASE_LENGTH + LENGTH_ROWS
BASE_LINE = BASE_COLUMN + COLUMN_ROWS
BASE_FROM_END = BASE_LINE + LINE_ROWS
BASE_BIAS = BASE_FROM_END + FROM_END_ROWS
TOTAL_ROWS = BASE_BIAS + 1  # 1575
FEATURE_COUNT = 10

_VOWELS = set("aeiou")


def skeleton(text: str, cls: int) -> str:
    """First letter plus the remaining non-vowel letters (lowercased); non-letter runs unchanged."""
    if cls != CLASS_LETTER:
        return text
    lowered = text.lower()
    return lowered[0] + "".join(c for c in lowered[1:] if c not in _VOWELS)


def featurize_tokens(tokens: list[Token]) -> list[list[int]]:
    total_lines = 1 + sum(1 for t in tokens if t.cls == CLASS_NEWLINE)
    rows: list[list[int]] = []
    line = 0
    column = 0
    for t in tokens:
        chars = list(t.text)  # code points, same as Array.from in JS
        rows.append(
            [
                BASE_WORD + hash_token(t.text, WORD_BUCKETS),
                BASE_SKELETON + hash_token(skeleton(t.text, t.cls), SKELETON_BUCKETS),
                BASE_SHAPE + t.shape,
                BASE_PREFIX + hash_token("".join(chars[:2]), AFFIX_BUCKETS),
                BASE_SUFFIX + hash_token("".join(chars[-2:]), AFFIX_BUCKETS),
                BASE_LENGTH + min(t.end - t.start, LENGTH_ROWS - 1),
                BASE_COLUMN + min(column, COLUMN_ROWS - 1),
                BASE_LINE + min(line, LINE_ROWS - 1),
                BASE_FROM_END + min(total_lines - 1 - line, FROM_END_ROWS - 1),
                BASE_BIAS,
            ]
        )
        if t.cls == CLASS_NEWLINE:
            line += 1
            column = 0
        else:
            column += 1
    return rows


def featurize(text: str) -> list[list[int]]:
    return featurize_tokens(tokenize(text))
