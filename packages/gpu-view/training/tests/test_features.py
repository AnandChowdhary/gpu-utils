import json
import re
from pathlib import Path

from gpu_view import features, match, timeres
from gpu_view.generate import ROLES, dataset
from gpu_view.lexicon import KEYWORDS
from gpu_view.schema import assert_disjoint

PACKAGE = Path(__file__).resolve().parents[2]


def test_one_row_per_model_token() -> None:
    schema = {"fields": [{"name": "status", "kind": "enum", "values": ["open"]}]}
    toks, rows = features.featurize("open issues, status open", schema)
    assert [t.text for t in toks] == ["open", "issues", ",", "status", "open"]
    assert len(rows) == len(toks)
    assert all(len(r) <= features.SLOTS for r in rows)
    assert all(0 <= v < features.FEATURE_ROWS for r in rows for v in r)


def test_lexicon_is_unique_and_mirrored() -> None:
    assert len(set(KEYWORDS)) == len(KEYWORDS)
    ts = (PACKAGE / "src" / "lexicon.ts").read_text()
    start = ts.index("KEYWORDS: readonly string[] = ") + len("KEYWORDS: readonly string[] = ")
    literal = ts[start : ts.index("];", start) + 1]
    # Biome may emit '"' with single quotes; read every string literal either way.
    found = [a if a is not None else b for a, b in
             ((m.group(1), m.group(2)) for m in re.finditer(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', literal))]
    found = [f.replace('\\"', '"').replace("\\\\", "\\") for f in found]
    assert found == KEYWORDS


def test_fixtures_match_committed_files() -> None:
    """The committed cross-language fixtures must be reproducible from this code."""
    cases = json.loads((PACKAGE / "test" / "fixtures" / "features.json").read_text())
    for c in cases:
        toks, rows = features.featurize(c["text"], c["schema"])
        assert [t.text for t in toks] == c["tokens"], c["text"]
        assert rows == c["rows"], c["text"]
    for c in json.loads((PACKAGE / "test" / "fixtures" / "time.json").read_text()):
        y, m, d = (int(x) for x in c["now"].split("-"))
        from datetime import date

        got = timeres.resolve(c["text"], date(y, m, d))
        assert (list(got) if got else None) == c["range"], c


def test_matcher_tiers() -> None:
    schema = {"fields": [
        {"name": "order_value", "kind": "number", "aliases": ["total"]},
        {"name": "createdAt", "kind": "date"},
        {"name": "status", "kind": "enum", "values": ["in progress", "done"]},
    ]}
    entries = match.field_entries(schema)
    assert match.resolve_words(["order", "value"], entries).quality == match.EXACT
    assert match.resolve_words(["totals"], entries).quality == match.STEM
    assert match.resolve_words(["tota"], entries).quality == match.PREFIX
    assert match.resolve_words(["statis"], entries).quality == match.TYPO
    assert match.resolve_words(["nothing"], entries) is None
    toks = match.model_tokens("in-progress items")
    spans = match.match_spans(toks, match.enum_entries(schema), enum=True)
    assert [(s.start, s.end, s.value) for s in spans] == [(0, 3, 0)]


def test_generator_invariants() -> None:
    assert_disjoint()
    for ex in dataset("train", 300, seed=123) + dataset("eval", 100, seed=124):
        assert len(ex.tokens) == len(ex.roles) == len(ex.boundaries)
        assert sum(ex.boundaries) >= 1
        assert all(0 <= r < len(ROLES) for r in ex.roles)
        # every boundary sits on a role token
        assert all(ex.roles[i] != 0 for i, b in enumerate(ex.boundaries) if b)
        toks, rows = features.featurize(ex.text, ex.schema)
        assert [t.text for t in toks] == ex.tokens
