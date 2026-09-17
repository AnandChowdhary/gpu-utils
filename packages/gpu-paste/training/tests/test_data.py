import numpy as np

from gpu_paste.data import LABELS, LEARNED_KINDS, SPAN_KINDS, Gen, generate
from gpu_paste.dataset import bio_labels, encode
from gpu_utils_training.features import tokenize


def test_generator_is_deterministic() -> None:
    a = generate(20, seed=5, offline=True)
    b = generate(20, seed=5, offline=True)
    assert [x.text for x in a] == [x.text for x in b]


def test_spans_are_within_text_and_aligned() -> None:
    g = Gen(11, offline=True)
    for _ in range(300):
        ex = g.example()
        n = len(ex.text.encode("utf-16-le")) // 2
        for s, e, kind in ex.spans:
            assert 0 <= s < e <= n
            assert kind in SPAN_KINDS
        assert ex.kind is None or ex.kind in LEARNED_KINDS


def test_bio_alignment() -> None:
    text = "Call Jane Doe on +1 555 0100 today."
    tokens = tokenize(text)
    spans = [(5, 13, "person"), (17, 28, "phone")]
    labels = bio_labels(text, tokens, spans)
    names = [LABELS[i] for i in labels]
    assert names[:5] == ["O", "O", "B-person", "I-person", "I-person"]
    assert "B-phone" in names
    enc = encode(type("E", (), {"text": text, "kind": "prose", "spans": spans})())
    assert enc.ids.shape == (len(tokens), 10)
    assert enc.kind == 2
