"""Tokenizer and sparse featurizer for gpu-cite. Must match src/features.ts byte-for-byte.

Every token gets a fixed-width row of ``WIDTH`` sparse ids into one embedding table.
Id 0 is the padding row (always a zero vector); real ids start at 1. Rows are:

    0  word hash (1024 buckets)          5  length bucket (16)
    1  consonant-skeleton hash (256)     6  relative position bucket (8)
    2  shape (8)                         7..11  up to five flag ids (lexicon / regex)
    3  first char hash (64)
    4  last char hash (64)

Flags are cue-word lexicon groups (``in``, ``pp.``, ``vol.``, ``eds.``, months, publisher
words, ...), token-form tests (year-like, initial, acronym, roman numeral) and "inside a
deterministic regex match" markers for URLs, DOIs and arXiv identifiers.
"""

from __future__ import annotations

import re

from gpu_utils_training.features import Token, hash_token, tokenize

WORD_BUCKETS = 1024
SKEL_BUCKETS = 256
SHAPES = 8
CHAR_BUCKETS = 64
LEN_BUCKETS = 16
POS_BUCKETS = 8
MAX_FLAGS = 5
WIDTH = 7 + MAX_FLAGS

OFF_WORD = 1
OFF_SKEL = OFF_WORD + WORD_BUCKETS
OFF_SHAPE = OFF_SKEL + SKEL_BUCKETS
OFF_FIRST = OFF_SHAPE + SHAPES
OFF_LAST = OFF_FIRST + CHAR_BUCKETS
OFF_LEN = OFF_LAST + CHAR_BUCKETS
OFF_POS = OFF_LEN + LEN_BUCKETS
OFF_FLAG = OFF_POS + POS_BUCKETS

FLAG_NAMES = [
    "YEARLIKE",
    "DIGITS_SHORT",
    "DIGITS_LONG",
    "INITIAL",
    "ACRONYM",
    "MONTH",
    "CUE_IN",
    "CUE_ED",
    "CUE_VOL",
    "CUE_NO",
    "CUE_PP",
    "CUE_PROC",
    "CUE_JOURNAL",
    "CUE_PUB",
    "CUE_UNIV",
    "CUE_THESIS",
    "CUE_REPORT",
    "CUE_ACCESS",
    "CUE_DOI",
    "CUE_ARXIV",
    "CUE_AND",
    "CUE_ETAL",
    "CUE_EDITION",
    "ORDINAL",
    "CITY",
    "PARTICLE",
    "SUFFIX",
    "IN_URL",
    "IN_DOI",
    "IN_ARXIV",
    "QUOTE",
    "BRACKET",
    "DASH",
    "CUE_WEB",
    "ROMAN",
    "CUE_PART",
    "STOP",
    "NONASCII",
]
FLAG = {name: i for i, name in enumerate(FLAG_NAMES)}
TABLE_ROWS = OFF_FLAG + len(FLAG_NAMES)

# Lexicon groups. Keep in sync with features.ts (same words, same groups).
LEXICON: dict[str, str] = {}


def _group(name: str, words: str) -> None:
    for w in words.split():
        LEXICON.setdefault(w, name)


_group("MONTH", "january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec")
_group("CUE_IN", "in")
_group("CUE_ED", "ed eds edited editor editors hrsg dir dirs")
_group("CUE_VOL", "vol volume vols bd jahrgang tome")
_group("CUE_NO", "no number issue num nr heft")
_group("CUE_PP", "pp p pages page pg pgs")
_group("CUE_PROC", "proceedings proc conference conf symposium symp workshop congress meeting annual international intl ieee acm usenix")
_group("CUE_JOURNAL", "journal j review rev letters lett transactions trans bulletin bull annals archives acta magazine quarterly studies science nature physics physical chemistry chemical biology medicine research reports communications comm")
_group("CUE_PUB", "press publishers publishing publications verlag books wiley springer elsevier routledge sage oxford cambridge mit pearson mcgraw hill penguin random house academic kluwer plenum prentice addison wesley blackwell macmillan palgrave harvard princeton yale stanford reilly siam nature")
_group("CUE_UNIV", "university univ universität universite college institute institut school department dept laboratory lab faculty")
_group("CUE_THESIS", "thesis dissertation phd ph msc ma master masters doctoral diss")
_group("CUE_REPORT", "report technical tech rep memo memorandum working paper tr rfc")
_group("CUE_ACCESS", "retrieved accessed available viewed visited cited online from")
_group("CUE_DOI", "doi")
_group("CUE_ARXIV", "arxiv preprint eprint biorxiv medrxiv ssrn hal abs")
_group("CUE_AND", "and & und et y e")
_group("CUE_ETAL", "al others")
_group("CUE_EDITION", "edition edn revised")
_group("ORDINAL", "st nd rd th")
_group("CITY", "new york london berlin cambridge oxford paris boston chicago heidelberg amsterdam tokyo beijing washington san francisco los angeles philadelphia dordrecht hoboken cham singapore sydney toronto delhi mumbai vienna zurich munich milan rome madrid barcelona stockholm copenhagen edinburgh dublin ny ma ca uk usa dc nj il pa")
_group("PARTICLE", "van von de der den la le di da del du bin ibn dos das")
_group("SUFFIX", "jr sr ii iii iv")
_group("CUE_WEB", "web website homepage blog wikipedia github http https www html htm com org net")
_group("CUE_PART", "chapter ch part sec section appendix")
_group("STOP", "of the a an for on to with by")

QUOTE_CHARS = set("\"'“”‘’«»‚„")
BRACKET_CHARS = set("()[]{}")
DASH_CHARS = set("-–—‐‑")
ROMAN_RE = re.compile(r"^(?:[ivxlc]{1,6})$")
YEAR_RE = re.compile(r"^(?:1[5-9][0-9][0-9]|20[0-9][0-9])$")

# Deterministic regexes. Written with explicit ASCII classes so JS and Python agree.
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'“”]+", re.IGNORECASE)
DOI_RE = re.compile(r"10\.[0-9]{4,9}/[^\s\"<>“”]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv[:\s]*|abs/|pdf/)([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)"
    r"|((?:astro-ph|hep-th|hep-ph|hep-ex|hep-lat|gr-qc|quant-ph|cond-mat|math-ph|nucl-th|nucl-ex"
    r"|chao-dyn|alg-geom|q-alg|solv-int|math|cs|physics|nlin|q-bio|q-fin|stat|econ|eess)"
    r"(?:\.[A-Za-z]{2})?/[0-9]{7}(?:v[0-9]+)?)",
    re.IGNORECASE,
)


def trim_match(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip trailing punctuation from a URL/DOI match, keeping balanced parentheses."""
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    if end > start and text[end - 1] == ")" and text[start:end].count("(") < text[start:end].count(")"):
        end -= 1
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    return start, end


def find_urls(text: str) -> list[tuple[int, int]]:
    return [trim_match(text, m.start(), m.end()) for m in URL_RE.finditer(text)]


def find_dois(text: str) -> list[tuple[int, int]]:
    return [trim_match(text, m.start(), m.end()) for m in DOI_RE.finditer(text)]


def find_arxiv(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in ARXIV_RE.finditer(text):
        g = 1 if m.group(1) is not None else 2
        out.append((m.start(g), m.end(g)))
    return out


def _utf16_offsets(text: str) -> list[int]:
    """Map Python str index -> UTF-16 offset, one entry per index plus a sentinel."""
    offs = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        offs[i] = acc
        acc += 2 if ord(ch) > 0xFFFF else 1
    offs[len(text)] = acc
    return offs


def skeleton(text: str) -> str:
    s = "".join(c for c in text.lower() if c not in "aeiou")
    return s if s else "_" + text.lower()


def token_flags(tok: Token, in_url: bool, in_doi: bool, in_arxiv: bool) -> list[int]:
    text = tok.text
    low = text.lower()
    flags: list[int] = []
    if tok.cls == 1:  # digits
        if YEAR_RE.match(text):
            flags.append(FLAG["YEARLIKE"])
        elif len(text) <= 3:
            flags.append(FLAG["DIGITS_SHORT"])
        elif len(text) >= 5:
            flags.append(FLAG["DIGITS_LONG"])
    elif tok.cls == 0:  # letters
        if len(text) == 1 and tok.shape == 1:
            flags.append(FLAG["INITIAL"])
        elif 2 <= len(text) <= 6 and tok.shape == 1:
            flags.append(FLAG["ACRONYM"])
        group = LEXICON.get(low)
        if group is not None:
            flags.append(FLAG[group])
        if ROMAN_RE.match(low) and group != "CUE_IN":
            flags.append(FLAG["ROMAN"])
        if any(ord(c) > 127 for c in text):
            flags.append(FLAG["NONASCII"])
    elif tok.cls == 4:  # other
        if text in QUOTE_CHARS:
            flags.append(FLAG["QUOTE"])
        elif text in BRACKET_CHARS:
            flags.append(FLAG["BRACKET"])
        elif text in DASH_CHARS:
            flags.append(FLAG["DASH"])
        elif text == "&":
            flags.append(FLAG["CUE_AND"])
    if in_url:
        flags.append(FLAG["IN_URL"])
    if in_doi:
        flags.append(FLAG["IN_DOI"])
    if in_arxiv:
        flags.append(FLAG["IN_ARXIV"])
    return flags[:MAX_FLAGS]


def _in_spans(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in spans)


def featurize_tokens(text: str, tokens: list[Token]) -> list[list[int]]:
    offs = _utf16_offsets(text)
    urls = [(offs[s], offs[e]) for s, e in find_urls(text)]
    dois = [(offs[s], offs[e]) for s, e in find_dois(text)]
    arx = [(offs[s], offs[e]) for s, e in find_arxiv(text)]
    n = len(tokens)
    rows: list[list[int]] = []
    for i, t in enumerate(tokens):
        row = [
            OFF_WORD + hash_token(t.text, WORD_BUCKETS),
            OFF_SKEL + hash_token(skeleton(t.text), SKEL_BUCKETS),
            OFF_SHAPE + t.shape,
            OFF_FIRST + hash_token(t.text[0], CHAR_BUCKETS),
            OFF_LAST + hash_token(t.text[-1], CHAR_BUCKETS),
            OFF_LEN + min(len(t.text), LEN_BUCKETS - 1),
            OFF_POS + (POS_BUCKETS * i) // n,
        ]
        flags = token_flags(
            t,
            _in_spans(urls, t.start, t.end),
            _in_spans(dois, t.start, t.end),
            _in_spans(arx, t.start, t.end),
        )
        row += [OFF_FLAG + f for f in flags]
        row += [0] * (WIDTH - len(row))
        rows.append(row)
    return rows


def featurize(text: str) -> list[list[int]]:
    return featurize_tokens(text, tokenize(text))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    cases = [
        "Smith, J., & Doe, A. B. (2019). A study of things. Journal of Stuff, 12(3), 45–67. https://doi.org/10.1000/xyz123",
        "[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning,” in Proc. CVPR, 2016, pp. 770–778.",
        "Vaswani A, et al. Attention is all you need. arXiv preprint arXiv:1706.03762v5, 2017.",
        "van der Berg, P. (n.d.) Über Café. Retrieved March 3, 2021, from www.example.org/x?y=1.",
        "Aynutdinov V. et al., Proc. 30th ICRC, Merida (Mexico) 2007; arXiv.org:astro-ph/0710.3063.",
        "Doe J. Title (2nd ed.). Vol. IV, pp. 12–34. Springer, 2001. doi: 10.1016/S0140-6736(20)30183-5.",
        "😀 emoji\ttab  double  space\r\nnewline",
    ]
    out = []
    for c in cases:
        toks = tokenize(c)
        out.append({"text": c, "tokens": [[t.text, t.start, t.end] for t in toks], "rows": featurize_tokens(c, toks),
                    "urls": find_urls(c), "dois": find_dois(c), "arxiv": find_arxiv(c)})
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "test" / "fixtures" / "features.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {target} ({len(out)} cases, table rows {TABLE_ROWS})")
