"""Tokenizer and sparse featurizer. Must match src/features.ts exactly."""

from __future__ import annotations

from gpu_utils_training.features import hash_token, tokenize  # shared reference impl


def featurize(text: str) -> list[list[int]]:
    return [[hash_token(t.text, 1024), t.shape, min(len(t.text), 15)] for t in tokenize(text)]
