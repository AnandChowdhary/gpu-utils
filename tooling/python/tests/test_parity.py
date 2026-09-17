import json
from pathlib import Path

from gpu_utils_training.features import hash_token, tokenize
from gpu_utils_training.quant import ALPHABET, encode, quantize
import numpy as np

FIXTURES = Path(__file__).resolve().parents[3] / "packages/runtime/test/fixtures/tokenize.json"


def test_tokenize_matches_fixtures() -> None:
    for case in json.loads(FIXTURES.read_text()):
        got = [[t.text, t.start, t.end, t.cls, t.shape] for t in tokenize(case["text"])]
        assert got == case["tokens"], case["text"]


def test_hash_matches_fixtures() -> None:
    for case in json.loads(FIXTURES.read_text()):
        for text, expected in case["hashes"].items():
            assert hash_token(text, 1024) == expected


def test_quant_roundtrip() -> None:
    t = np.array([0.5, -0.25, 0.0, 1.0], dtype=np.float32)
    q, scale = quantize(t)
    assert q.tolist() == [16, -8, 0, 31]
    assert encode(q) == ALPHABET[48] + ALPHABET[24] + ALPHABET[32] + ALPHABET[63]
