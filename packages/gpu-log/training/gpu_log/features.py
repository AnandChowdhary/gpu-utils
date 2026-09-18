"""Tokenizer and sparse featurizer for one log line. Must match src/features.ts exactly.

Every token gets FEATURE_COUNT sparse ids, each already offset into one flat embedding
table (see FEATURE_SIZES). Parity with TypeScript is enforced by model/fixtures.json.
"""

from __future__ import annotations

import unicodedata

from gpu_utils_training.features import Token, hash_token, tokenize

WORD_BUCKETS = 1024
PREFIX_BUCKETS = 512

# (name, rows). Order matters: it defines the id layout of the embedding table.
FEATURE_SIZES: list[tuple[str, int]] = [
    ("word", WORD_BUCKETS),
    ("prefix", PREFIX_BUCKETS),
    ("shape", 8),
    ("first", 129),
    ("last", 129),
    ("len", 16),
    ("pos", 20),
    ("rpos", 12),
    ("col", 10),
]
FEATURE_COUNT = len(FEATURE_SIZES)
FEATURE_OFFSETS: list[int] = []
_acc = 0
for _name, _size in FEATURE_SIZES:
    FEATURE_OFFSETS.append(_acc)
    _acc += _size
EMBED_ROWS = _acc


def _char_bucket(ch: str) -> int:
    """ASCII code, non-ASCII -> 128, decimal digits collapsed to '0' (like hash_token)."""
    if unicodedata.category(ch) == "Nd":
        return 48
    code = ord(ch)
    return code if code < 128 else 128


def _pos_bucket(i: int) -> int:
    if i < 16:
        return i
    if i < 32:
        return 16
    if i < 64:
        return 17
    if i < 128:
        return 18
    return 19


def _rpos_bucket(r: int) -> int:
    if r < 8:
        return r
    if r < 16:
        return 8
    if r < 32:
        return 9
    if r < 64:
        return 10
    return 11


def _col_bucket(c: int) -> int:
    if c == 0:
        return 0
    if c <= 8:
        return 1
    if c <= 16:
        return 2
    if c <= 24:
        return 3
    if c <= 32:
        return 4
    if c <= 48:
        return 5
    if c <= 64:
        return 6
    if c <= 96:
        return 7
    if c <= 160:
        return 8
    return 9


def token_features(tokens: list[Token], i: int) -> list[int]:
    t = tokens[i]
    text = t.text
    n = len(tokens)
    raw = [
        hash_token(text, WORD_BUCKETS),
        hash_token(text[:3], PREFIX_BUCKETS),
        t.shape,
        _char_bucket(text[0]),
        _char_bucket(text[-1]),
        min(len(text), 15),
        _pos_bucket(i),
        _rpos_bucket(n - 1 - i),
        _col_bucket(t.start),
    ]
    return [FEATURE_OFFSETS[k] + v for k, v in enumerate(raw)]


def featurize(line: str) -> tuple[list[Token], list[list[int]]]:
    """Tokenize one line (no newlines) and return (tokens, rows)."""
    tokens = tokenize(line)
    return tokens, [token_features(tokens, i) for i in range(len(tokens))]
