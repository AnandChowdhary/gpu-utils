import collections

from gpu_log import data
from gpu_log.gen import KINDS, ROLES


def test_generator_is_deterministic_and_aligned() -> None:
    a = data.generate(400, 3)
    b = data.generate(400, 3)
    assert a == b
    _, misaligned = data.build(a)
    assert misaligned == 0


def test_markup_round_trip() -> None:
    for ex in data.generate(300, 4):
        assert data.parse_markup(data.to_markup(ex)) == ex


def test_generator_covers_all_roles_and_kinds() -> None:
    exs = data.generate(3000, 5)
    roles = collections.Counter(r for ex in exs for _, _, r in ex.spans)
    kinds = collections.Counter(ex.kind for ex in exs)
    assert set(roles) == set(ROLES)
    assert set(kinds) == set(KINDS)


def test_unfamiliar_set_is_large_and_aligned() -> None:
    exs = data.read_markup_file(data.DATA_DIR / "unfamiliar.txt")
    assert len(exs) >= 60
    _, misaligned = data.build(exs)
    assert misaligned == 0


def test_bio_tags_are_consistent() -> None:
    for ex in data.generate(200, 6):
        _, tags, _, _ = data.tag_example(ex)
        prev = "O"
        for t in tags:
            lab = data.LABELS[t]
            if lab.startswith("I-"):
                assert prev[2:] == lab[2:], (ex.text, prev, lab)
            prev = lab
