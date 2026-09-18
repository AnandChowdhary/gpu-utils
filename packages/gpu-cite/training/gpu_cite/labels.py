"""Label inventories shared by the renderer, the model, the exporter and the evaluator.

Mirrors the constants in src/decode.ts; the exported manifest carries them too so the
TypeScript side never hard-codes indices.
"""

from __future__ import annotations

ROLES = [
    "AUTHOR",
    "TITLE",
    "CONTAINER",
    "YEAR",
    "VOLUME",
    "ISSUE",
    "PAGES",
    "PUBLISHER",
    "LOCATION",
    "EDITION",
    "DOI",
    "ARXIV",
    "URL",
    "ACCESSED",
    "EDITOR",
]
TAGS = ["O"] + [f"B-{r}" for r in ROLES] + [f"I-{r}" for r in ROLES]
TAG_INDEX = {t: i for i, t in enumerate(TAGS)}
TYPES = ["article", "book", "chapter", "conference", "thesis", "report", "web", "preprint"]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
NAMEPARTS = ["O", "GIVEN", "FAMILY"]


def b_tag(role: str) -> int:
    return TAG_INDEX[f"B-{role}"]


def i_tag(role: str) -> int:
    return TAG_INDEX[f"I-{role}"]


def tag_role(tag: int) -> str | None:
    if tag == 0:
        return None
    return TAGS[tag][2:]


def is_begin(tag: int) -> bool:
    return 0 < tag <= len(ROLES)


def spans_from_tags(tags: list[int], tokens: list) -> list[tuple[str, int, int]]:
    """Group BIO tags into (role, start_char, end_char) entity spans (UTF-16 offsets)."""
    out: list[tuple[str, int, int]] = []
    cur: tuple[str, int, int] | None = None
    for tag, tok in zip(tags, tokens, strict=True):
        role = tag_role(tag)
        if role is None:
            if cur:
                out.append(cur)
                cur = None
            continue
        if is_begin(tag) or cur is None or cur[0] != role:
            if cur:
                out.append(cur)
            cur = (role, tok.start, tok.end)
        else:
            cur = (role, cur[1], tok.end)
    if cur:
        out.append(cur)
    return out
