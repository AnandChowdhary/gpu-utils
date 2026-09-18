import json
import re
from pathlib import Path

from gpu_email import features as F
from gpu_email.features import NUM_ROWS, NUM_SLOTS, SLOT_BASE, SLOTS, featurize, line_infos
from gpu_utils_training.features import tokenize

PKG = Path(__file__).resolve().parents[2]


def test_one_row_per_token_fixed_width() -> None:
    rows = featurize("hello world")
    assert len(rows) == 3  # hello, space, world
    assert all(len(r) == NUM_SLOTS for r in rows)


def test_ids_stay_inside_their_slot() -> None:
    text = "Hi Bob,\n\nThanks!\n-- \nJohn Doe\nCEO | Acme\n+1 555 123 4567\njohn@acme.com\n> quoted\nOn Mon, Jan 5, 2024 Bob wrote:\n"
    for row in featurize(text):
        for s, v in enumerate(row):
            assert SLOT_BASE[s] <= v < SLOT_BASE[s] + SLOTS[s][1]
            assert v < NUM_ROWS


def test_line_infos_rules() -> None:
    tokens = tokenize("> quoted\n-- \nsig\nFrom: x\n\nOn Mon Bob wrote:\n")
    lines = line_infos(tokens)
    assert [ln.quote_prefixed for ln in lines] == [True, False, False, False, False, False]
    assert [ln.delimiter for ln in lines] == [False, True, False, False, False, False]
    assert lines[3].is_header and not lines[2].is_header
    assert lines[4].blank
    assert lines[5].is_attrib_marker
    assert lines[1].char_start == 9 and lines[1].char_end == 12


def _parse(ts_array: str):
    """Biome formats the generated arrays with trailing commas; normalise before json.loads."""
    return json.loads(re.sub(r",(\s*[\]\}])", r"\1", ts_array))


def test_generated_keywords_ts_is_current() -> None:
    """src/keywords.ts is generated from features.py; fail if it drifted."""
    ts = (PKG / "src" / "keywords.ts").read_text()
    for name in ["KW_WROTE", "KW_HEADER", "KW_DISCLAIMER", "KW_CLOSING_WORDS", "KW_GREETING_WORDS", "KW_DATE_WORDS", "KW_MOBILE", "KW_FORWARD", "KW_ORIGINAL", "KW_CONTACT_WORDS", "KW_CLOSING_PHRASES", "KW_GREETING_PHRASES", "EDGES_LINE_IDX", "LINE_KINDS", "BIO_LABELS"]:
        m = re.search(rf"export const {name} = (\[.*?\]) as const;", ts, re.S)
        assert m, name
        assert _parse(m.group(1)) == list(getattr(F, name)), name
    m = re.search(r"export const SLOTS = (\[.*?\]) as const;", ts, re.S)
    assert m and _parse(m.group(1)) == [[n, s] for n, s in SLOTS]
    assert f"export const NUM_ROWS = {NUM_ROWS};" in ts
    assert f"export const NUM_SLOTS = {NUM_SLOTS};" in ts


def test_utf16_offsets_for_astral_chars() -> None:
    rows = featurize("📞 +1 555 123 4567\nJohn\n")
    tokens = tokenize("📞 +1 555 123 4567\nJohn\n")
    assert tokens[0].end == 2  # surrogate pair
    assert len(rows) == len(tokens)
