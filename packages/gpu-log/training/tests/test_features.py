import json
from pathlib import Path

from gpu_log.features import EMBED_ROWS, FEATURE_COUNT, FEATURE_OFFSETS, featurize

FIXTURES = Path(__file__).resolve().parents[2] / "model" / "fixtures.json"


def test_one_row_per_token() -> None:
    tokens, rows = featurize("hello world")
    assert len(tokens) == 3
    assert len(rows) == 3
    assert all(len(r) == FEATURE_COUNT for r in rows)


def test_ids_stay_within_their_tables() -> None:
    _, rows = featurize("2024-01-15T10:30:00Z INFO app: café 🚀 done k=v")
    bounds = FEATURE_OFFSETS + [EMBED_ROWS]
    for row in rows:
        for k, v in enumerate(row):
            assert bounds[k] <= v < bounds[k + 1]


def test_fixtures_match_featurizer() -> None:
    if not FIXTURES.exists():
        return
    for case in json.loads(FIXTURES.read_text()):
        assert featurize(case["text"])[1] == case["features"], case["text"]
