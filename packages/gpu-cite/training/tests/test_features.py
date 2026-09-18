import json
from pathlib import Path

from gpu_cite.features import TABLE_ROWS, WIDTH, featurize, featurize_tokens, find_arxiv, find_dois, find_urls
from gpu_utils_training.features import tokenize

FIXTURES = Path(__file__).resolve().parents[1].parent / "test" / "fixtures" / "features.json"


def test_one_row_per_token() -> None:
    rows = featurize("hello world")
    assert len(rows) == 3
    assert all(len(r) == WIDTH for r in rows)
    assert all(0 <= v < TABLE_ROWS for r in rows for v in r)


def test_matches_committed_fixture() -> None:
    """The JSON fixture is what the TypeScript side is tested against; it must not drift."""
    for case in json.loads(FIXTURES.read_text(encoding="utf-8")):
        toks = tokenize(case["text"])
        assert [[t.text, t.start, t.end] for t in toks] == case["tokens"]
        assert featurize_tokens(case["text"], toks) == case["rows"]
        assert [list(x) for x in find_urls(case["text"])] == case["urls"]
        assert [list(x) for x in find_dois(case["text"])] == case["dois"]
        assert [list(x) for x in find_arxiv(case["text"])] == case["arxiv"]


def test_identifier_regexes() -> None:
    s = "doi: 10.1016/S0140-6736(20)30925-9. PMID"
    (a, b), = find_dois(s)
    assert s[a:b] == "10.1016/S0140-6736(20)30925-9"
    s = "arXiv:1706.03762v5 and hep-ph/9911415 and 2020.12345 alone"
    assert [s[a:b] for a, b in find_arxiv(s)] == ["1706.03762v5", "hep-ph/9911415"]
    s = "see www.example.org/x?y=1."
    assert [s[a:b] for a, b in find_urls(s)] == ["www.example.org/x?y=1"]


def test_flags_are_position_aware() -> None:
    rows = featurize("A B C D E F G H")
    assert rows[0][6] != rows[-1][6]  # position bucket differs between first and last token
