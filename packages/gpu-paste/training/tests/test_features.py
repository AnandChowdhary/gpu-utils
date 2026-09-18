import json
from pathlib import Path

from gpu_paste.features import FEATURE_COUNT, TOTAL_ROWS, featurize

FIXTURES = Path(__file__).resolve().parents[2] / "model" / "fixtures.json"


def test_one_row_per_token() -> None:
    rows = featurize("hello world")
    assert len(rows) == 3  # hello, space, world
    assert all(len(r) == FEATURE_COUNT for r in rows)
    assert all(0 <= v < TOTAL_ROWS for r in rows for v in r)


def test_line_and_column_features() -> None:
    rows = featurize("a b\nc")
    # column ids: a=0, space=1, b=2, newline=3, c=0 (relative to the COLUMN base)
    cols = [r[6] - rows[0][6] for r in rows]
    assert cols == [0, 1, 2, 3, 0]
    lines = [r[7] - rows[0][7] for r in rows]
    assert lines == [0, 0, 0, 0, 1]


def test_fixtures_roundtrip() -> None:
    """The exported canonical fixtures must reproduce from the current featurizer."""
    if not FIXTURES.exists():
        return
    for case in json.loads(FIXTURES.read_text())["cases"]:
        assert featurize(case["input"]) == case["rows"], case["input"]
