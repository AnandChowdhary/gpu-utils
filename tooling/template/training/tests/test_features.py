from __SNAKE__.features import FEATURE_ROWS, SLOTS, featurize


def test_one_row_per_token() -> None:
    rows = featurize("hello world")
    assert len(rows) == 3  # hello, space, world
    assert all(len(r) == SLOTS for r in rows)
    assert all(0 <= v < FEATURE_ROWS for r in rows for v in r)
