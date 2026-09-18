"""Real, human-labelled evaluation sets converted into gpu-cite's span format.

- anystyle core (BSD-2-Clause): ``<sequence>`` of tagged segments joined by spaces.
- GROBID citation corpus (Apache-2.0): TEI ``<bibl>`` mixed content.
- The hand-written "unfamiliar" set (``data/unfamiliar.jsonl``): fields as strings which
  are located in the text to make spans.

Each case is ``{"text", "spans": {role: [[start, end], ...]}, "type"?: str, "authors"?: [...]}``
with UTF-16 offsets. Both corpora are used for evaluation only.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .labels import ROLES
from .metrics import _utf16, normalize_span
from .sources import CACHE, fetch_anystyle_core, fetch_grobid_citations

DATA = Path(__file__).resolve().parents[1] / "data"

ANYSTYLE_MAP = {
    "author": "AUTHOR",
    "title": "TITLE",
    "journal": "CONTAINER",
    "container-title": "CONTAINER",
    "date": "YEAR",
    "volume": "VOLUME",
    "pages": "PAGES",
    "publisher": "PUBLISHER",
    "location": "LOCATION",
    "editor": "EDITOR",
    "url": "URL",
    "doi": "DOI",
    "edition": "EDITION",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()


def _add_span(case: dict[str, Any], role: str, start: int, end: int) -> None:
    t16 = _utf16(case["text"])
    s, e = normalize_span(t16, start, end)
    if s < e:
        case["spans"].setdefault(role, []).append([s, e])


def load_anystyle(limit: int | None = None) -> list[dict[str, Any]]:
    path = fetch_anystyle_core()
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for seq in root.findall("sequence"):
        text = ""
        segs: list[tuple[str, int, int]] = []
        for el in seq:
            seg = _clean(el.text or "")
            if not seg:
                continue
            if text:
                text += " "
            start = len(text.encode("utf-16-le")) // 2
            text += seg
            end = len(text.encode("utf-16-le")) // 2
            segs.append((el.tag, start, end))
        case: dict[str, Any] = {"text": text, "spans": {}, "source": "anystyle"}
        for tag, s, e in segs:
            role = ANYSTYLE_MAP.get(tag)
            if role:
                _add_span(case, role, s, e)
        # anystyle splits "volume" and "issue" inconsistently; issue lives inside volume.
        if case["spans"]:
            cases.append(case)
        if limit and len(cases) >= limit:
            break
    return cases


def _grobid_bibl(bibl: ET.Element) -> dict[str, Any] | None:
    ns = "{http://www.tei-c.org/ns/1.0}"
    text = ""
    spans: list[tuple[str, int, int]] = []
    has_a = any(el.tag == f"{ns}title" and el.get("level") == "a" for el in bibl)
    has_j = any(el.tag == f"{ns}title" and el.get("level") == "j" for el in bibl)

    def role_of(el: ET.Element) -> str | None:
        tag = el.tag.replace(ns, "")
        if tag == "author":
            return "AUTHOR"
        if tag == "editor":
            return "EDITOR"
        if tag == "title":
            lvl = el.get("level")
            if lvl == "a":
                return "TITLE"
            if lvl == "j":
                return "CONTAINER"
            if lvl in ("m", "s"):
                return "CONTAINER" if (has_a or has_j) else "TITLE"
            return "TITLE"
        if tag == "date":
            return "YEAR"
        if tag == "biblScope":
            unit = el.get("unit")
            return {"volume": "VOLUME", "issue": "ISSUE", "page": "PAGES"}.get(unit or "")
        if tag == "publisher":
            return "PUBLISHER"
        if tag == "pubPlace":
            return "LOCATION"
        if tag == "idno":
            t = (el.get("type") or "").lower()
            return {"doi": "DOI", "arxiv": "ARXIV", "url": "URL"}.get(t)
        if tag == "ptr":
            return "URL"
        return None

    def append(s: str) -> None:
        nonlocal text
        text += s

    append(_clean(bibl.text or "") + (" " if bibl.text and bibl.text.strip() else ""))
    for el in bibl:
        inner = _clean("".join(el.itertext()))
        role = role_of(el)
        if inner:
            if text and not text.endswith(" ") and not text.endswith("(") and text[-1] not in "“\"'":
                append(" " if el.tag.replace(ns, "") not in ("biblScope", "date") or text.endswith(",") else "")
            start = len(text.encode("utf-16-le")) // 2
            append(inner)
            end = len(text.encode("utf-16-le")) // 2
            if role:
                spans.append((role, start, end))
        tail = _clean(el.tail or "")
        if tail:
            if tail[0].isalnum() or tail[0] in "([":
                append(" ")
            append(tail)
            append(" " if el.tail and el.tail.endswith((" ", "\n", "\t")) else "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 15 or not spans:
        return None
    case: dict[str, Any] = {"text": text, "spans": {}, "source": "grobid"}
    # Recompute spans on the whitespace-normalised text by locating the segment text.
    cursor = 0
    for role, s, e in spans:
        seg = None
        # original substring is stable because we only collapsed runs of whitespace
        # (already single) and stripped edges; search from cursor for robustness.
        for r2, s2, e2 in [(role, s, e)]:
            seg = _clean(text[s2:e2]) if s2 < len(text) else None
        if not seg:
            continue
        pos = text.find(seg, max(0, cursor - 2))
        if pos < 0:
            pos = text.find(seg)
        if pos < 0:
            continue
        _add_span(case, role, pos, pos + len(seg))
        cursor = pos + len(seg)
    return case if case["spans"] else None


def load_grobid(limit: int | None = None) -> list[dict[str, Any]]:
    path = fetch_grobid_citations()
    raw = path.read_text(encoding="utf-8")
    cases: list[dict[str, Any]] = []
    for file_xml in re.findall(r"<file path=\"[^\"]*\">(.*?)</file>", raw, flags=re.S):
        try:
            root = ET.fromstring(file_xml.strip())
        except ET.ParseError:
            continue
        for bibl in root.iter("{http://www.tei-c.org/ns/1.0}bibl"):
            case = _grobid_bibl(bibl)
            if case:
                cases.append(case)
            if limit and len(cases) >= limit:
                return cases
    return cases


def load_unfamiliar() -> list[dict[str, Any]]:
    """Hand-written references (data/unfamiliar.jsonl) with string-valued fields."""
    cases: list[dict[str, Any]] = []
    for line in (DATA / "unfamiliar.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = json.loads(line)
        text = row["text"]
        t16 = _utf16(text)
        case: dict[str, Any] = {"text": text, "spans": {}, "type": row.get("type"), "source": "unfamiliar", "fields": row}
        for role in ROLES:
            key = role.lower()
            if key == "author":
                key = "authors"
            if key == "editor":
                key = "editors"
            vals = row.get(key)
            if vals is None:
                continue
            if isinstance(vals, (str, int)):
                vals = [str(vals)]
            elif isinstance(vals, dict):
                vals = [vals["text"]]
            elif isinstance(vals, list) and vals and isinstance(vals[0], dict):
                vals = [v["text"] for v in vals]
            cursor = 0
            found: list[tuple[int, int]] = []
            for v in vals:
                pos = t16.find(v, cursor)
                if pos < 0:
                    pos = t16.find(v)
                if pos < 0:
                    raise ValueError(f"field value {v!r} not in text {text!r}")
                found.append((pos, pos + len(v)))
                cursor = pos + len(v)
            if role in ("AUTHOR", "EDITOR"):
                _add_span(case, role, min(s for s, _ in found), max(e for _, e in found))
                case[f"{role.lower()}_entities"] = [list(normalize_span(t16, s, e)) for s, e in found]
            else:
                for s, e in found:
                    _add_span(case, role, s, e)
        cases.append(case)
    return cases


if __name__ == "__main__":
    a = load_anystyle()
    g = load_grobid()
    print(f"anystyle {len(a)} cases, grobid {len(g)} cases")
    for c in a[:3] + g[:3]:
        print(c["text"])
        for role, sp in c["spans"].items():
            print("  ", role, [c["text"][s:e] for s, e in sp])
    u = load_unfamiliar()
    print(f"unfamiliar {len(u)} cases")
