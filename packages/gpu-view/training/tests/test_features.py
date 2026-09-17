from gpu_view.features import featurize


def test_one_row_per_token() -> None:
    assert len(featurize("hello world")) == 3  # hello, space, world
