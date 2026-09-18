import random

from gpu_cite.data import STYLES, Pools, realize, render
from gpu_cite.labels import ROLES, TAGS, TYPES, spans_from_tags
from gpu_utils_training.features import tokenize

RECORDS = [
    {
        "source": "crossref", "type": "article", "title": "A study of things and other matters",
        "authors": [{"given": "John A.", "family": "Smith"}, {"given": "Zeynab", "family": "Sadeghi"}, {"given": "Yvette P.", "family": "Geels"}],
        "year": 2019, "doi": "10.1000/xyz123", "container": "Journal of Interesting Stuff", "container_short": "J. Int. Stuff",
        "volume": "12", "issue": "3", "pages": {"from": "45", "to": "67"}, "publisher": "Elsevier BV",
    },
    {
        "source": "crossref", "type": "chapter", "title": "Inductive learning of disjointness axioms",
        "authors": [{"given": "Daniel", "family": "Fleischhacker"}], "year": 2011, "doi": "10.1007/978-3-642-25106-1_20",
        "container": "Lecture Notes in Computer Science", "pages": {"from": "680", "to": "697"}, "publisher": "Springer",
        "editors": [{"given": "Anna", "family": "Weber"}],
    },
    {
        "source": "arxiv", "type": "preprint", "title": "Attention is all you need",
        "authors": [{"given": "Ashish", "family": "Vaswani"}, {"given": "Noam", "family": "Shazeer"}], "year": 2017,
        "arxiv": "1706.03762", "month": 6, "category": "cs.CL",
    },
    {
        "source": "crossref", "type": "conference", "title": "Deep residual learning for image recognition",
        "authors": [{"given": "Kaiming", "family": "He"}, {"given": "Xiangyu", "family": "Zhang"}], "year": 2016,
        "doi": "10.1109/cvpr.2016.90", "container": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        "pages": {"from": "770", "to": "778"}, "publisher": "IEEE",
    },
]


def _valid_bio(tags: list[int]) -> bool:
    R = len(ROLES)
    prev = 0
    for t in tags:
        if t > R:  # I-X must follow B-X or I-X of the same role
            role = (t - 1) % R
            if prev == 0 or (prev - 1) % R != role:
                return False
        prev = t
    return True


def test_every_style_renders_every_type_with_consistent_labels() -> None:
    rng = random.Random(0)
    pools = Pools(RECORDS * 10, rng)
    seen_types: set[int] = set()
    for style_name, _, _ in STYLES:
        for _ in range(40):
            rec = realize(rng.choice(RECORDS), rng, pools)
            if rec is None:
                continue
            ex = render(rec, rng, style_name)
            toks = tokenize(ex["text"])
            assert len(toks) == len(ex["tags"]) == len(ex["parts"]), ex["text"]
            assert _valid_bio(ex["tags"]), (style_name, ex["text"], ex["tags"])
            assert 0 <= ex["type"] < len(TYPES)
            seen_types.add(ex["type"])
            # name parts only inside author/editor spans
            for tag, part in zip(ex["tags"], ex["parts"], strict=True):
                if part:
                    assert TAGS[tag][2:] in ("AUTHOR", "EDITOR")
            # every field span reconstructs to non-empty text
            for role, s, e in spans_from_tags(ex["tags"], toks):
                assert role in ROLES
                assert ex["text"][s:e].strip(), (role, ex["text"])
    assert len(seen_types) == len(TYPES)


def test_render_is_deterministic() -> None:
    a = render(realize(RECORDS[0], random.Random(1), Pools(RECORDS, random.Random(1))), random.Random(7), "apa")
    b = render(realize(RECORDS[0], random.Random(1), Pools(RECORDS, random.Random(1))), random.Random(7), "apa")
    assert a == b


def test_identifiers_survive_rendering() -> None:
    rng = random.Random(3)
    pools = Pools(RECORDS, rng)
    hits = 0
    for _ in range(200):
        rec = realize(RECORDS[0], rng, pools)
        if rec is None or not rec.get("doi"):
            continue
        ex = render(rec, rng)
        if "10.1000/xyz123" in ex["text"]:
            hits += 1
            toks = tokenize(ex["text"])
            dois = [ex["text"][s:e] for r, s, e in spans_from_tags(ex["tags"], toks) if r == "DOI"]
            assert dois == ["10.1000/xyz123"], ex["text"]
    assert hits > 20
