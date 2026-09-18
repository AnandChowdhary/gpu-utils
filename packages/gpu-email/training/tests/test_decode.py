import numpy as np
from gpu_utils_training.decode import viterbi
from gpu_utils_training.features import tokenize

from gpu_email.data import EmailGen, encode_example, generate, label_tokens, render
from gpu_email.decode import BIO_T, decode
from gpu_email.features import BIO_LABELS, LINE_KINDS, featurize_tokens, line_infos

K = len(LINE_KINDS)
B = len(BIO_LABELS)


def _logits_from_labels(kinds: list[int], bio: np.ndarray, T: int) -> np.ndarray:
    """Oracle logits: +6 on the gold class per token, 0 elsewhere."""
    lg = np.zeros((T, K + B), dtype=np.float32)
    for i in range(T):
        if kinds[i] >= 0:
            lg[i, kinds[i]] = 6.0
        lg[i, K + int(bio[i])] = 6.0
    return lg


def test_oracle_logits_roundtrip_generated_email() -> None:
    """With perfect token logits the decoder reproduces the generator's labels."""
    gen = EmailGen(11)
    checked = 0
    for _ in range(40):
        lines, meta = gen.email()
        ex = render(lines, gen)
        tokens, kinds, bio = label_tokens(ex)
        infos = line_infos(tokens)
        logits = _logits_from_labels(kinds.tolist(), bio, len(tokens))
        out = decode(tokens, infos, logits, ex.text)
        gold = [k for k in ex.line_kinds[: len(infos)]]
        pred = out["line_kinds"]
        # rules ('>' → quote, '--' → signature, everything below '--' → signature)
        # may legitimately override a generated label
        after_delim = False
        for g, p, ln in zip(gold, pred, infos):
            after_delim = after_delim or ln.delimiter
            if g < 0 or ln.quote_prefixed or after_delim:
                continue
            assert g == p, (meta, ln.text)
        assert out["reply"] == ex.reply or any(ln.delimiter for ln in infos)
        if ex.contact and not any(ln.delimiter for ln in infos):
            for key, val in ex.contact.items():
                assert out["contact"] is not None and out["contact"].get(key) == val, (key, val, out["contact"])
            checked += 1
    assert checked > 5


def test_viterbi_forbids_illegal_bio_transitions() -> None:
    em = np.zeros((3, B))
    em[0, BIO_LABELS.index("O")] = 1
    em[1, BIO_LABELS.index("I-NAME")] = 5  # I without B must be avoided
    em[2, BIO_LABELS.index("I-NAME")] = 5
    path = viterbi(em, BIO_T)
    labels = [BIO_LABELS[p] for p in path]
    assert "I-NAME" not in labels[:1]
    assert labels[1] != "I-NAME" or labels[0] in ("B-NAME", "I-NAME")


def test_exact_rules_and_regexes() -> None:
    text = "reply line\n> quoted\n-- \nJane\njane@example.com\nhttps://example.com/jane.\n"
    tokens = tokenize(text)
    infos = line_infos(tokens)
    logits = np.zeros((len(tokens), K + B), dtype=np.float32)
    logits[:, 0] = 3.0  # model says "reply" everywhere; rules must win
    out = decode(tokens, infos, logits, text)
    assert [LINE_KINDS[k] for k in out["line_kinds"]] == ["reply", "quote", "signature", "signature", "signature", "signature"]
    assert out["reply"] == "reply line"
    assert out["contact"]["email"] == ["jane@example.com"]
    assert out["contact"]["url"] == ["https://example.com/jane"]


def test_encode_example_alignment() -> None:
    ex = generate(3, 5)[0]
    enc = encode_example(ex)
    assert enc.rows.shape[0] == len(enc.kinds) == len(enc.bio)
    assert set(np.unique(enc.bio)) <= set(range(B))
    rows = featurize_tokens(tokenize(ex.text))
    assert np.array_equal(np.asarray(rows, dtype=np.int16), enc.rows)
