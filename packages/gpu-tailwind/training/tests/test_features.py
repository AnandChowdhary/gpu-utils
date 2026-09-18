import json
from pathlib import Path

from gpu_tailwind.features import ROWS, WIDTH, featurize, skeleton
from gpu_tailwind.lexicon import resolve_variant, words_of
from gpu_tailwind.pairing import compile_pieces
from gpu_tailwind.semantics import emit, parse_value

FIXTURES = Path(__file__).resolve().parents[2] / "model" / "fixtures.json"


def test_one_row_per_token() -> None:
    rows = featurize("hello world")
    assert len(rows) == 3  # hello, space, world
    assert all(len(r) == WIDTH for r in rows)
    assert all(0 <= i < ROWS for r in rows for i in r)


def test_skeleton() -> None:
    assert skeleton("shadow") == "shdw"
    assert skeleton("rounded") == "rnd"
    assert skeleton("Hello") == "hl"
    assert skeleton("16px") == "16px"


def test_features_match_exported_fixtures() -> None:
    if not FIXTURES.exists():
        return
    fixtures = json.loads(FIXTURES.read_text())
    assert len(fixtures) >= 20
    for case in fixtures:
        assert featurize(case["text"]) == case["features"], case["text"]


def test_semantics_examples() -> None:
    assert emit("shadow", parse_value("subtle"), False) == ["shadow-sm"]
    assert emit("rounded", parse_value("pill"), False) == ["rounded-full"]
    assert emit("text", parse_value("bold"), False) == ["font-bold"]
    assert emit("p", parse_value("16px"), False) == ["p-[16px]"]
    assert emit("w", parse_value("half"), False) == ["w-1/2"]
    assert emit("border", parse_value("32"), False) == []
    assert resolve_variant(words_of("on mobile")) == "max-sm"
    assert resolve_variant(words_of("when the parent is hovered")) == "group-hover"


def test_pairing_mirror() -> None:
    pieces = [("VAL", "bold", None), ("VAL", "red", None), ("PROP", "text", "text")]
    assert compile_pieces(pieces) == ["font-bold", "text-red-500"]
    pieces = [("VAL", "blue", None), ("VAR", "on hover", None)]
    assert compile_pieces(pieces) == ["hover:bg-blue-500"]
    pieces = [("NEG", "no", None), ("PROP", "shadow", "shadow")]
    assert compile_pieces(pieces) == ["shadow-none"]
