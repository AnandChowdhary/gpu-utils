"""Structured bibliographic metadata sources for the synthetic renderer.

Everything here is downloaded on demand into ``training/data/cache`` (gitignored) and
never committed:

- CrossRef REST API ``/works?sample=100`` (metadata is CC0 / public domain per CrossRef's
  terms; polite pool via ``mailto``).
- arXiv OAI-PMH ``ListRecords`` with the ``arXiv`` metadata prefix (arXiv metadata is CC0).
- anystyle ``res/parser/core.xml`` (BSD-2-Clause) – a human-labelled reference corpus used
  for evaluation only.

Run ``uv run python -m gpu_cite.sources`` to warm the cache.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
MAILTO = "gpu-utils@anandchowdhary.com"
USER_AGENT = f"gpu-cite-training/0.1 (https://github.com/AnandChowdhary/gpu-utils; mailto:{MAILTO})"

CROSSREF_TYPES = {
    "journal-article": "article",
    "book-chapter": "chapter",
    "proceedings-article": "conference",
    "book": "book",
    "monograph": "book",
    "edited-book": "book",
    "reference-book": "book",
    "report": "report",
    "report-component": "report",
    "dissertation": "thesis",
    "posted-content": "preprint",
}


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 – retry on any network error
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


# ----------------------------------------------------------------------------- CrossRef


def _clean_title(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)  # strip inline markup (<i>, <sub>, ...)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _crossref_record(item: dict[str, Any]) -> dict[str, Any] | None:
    kind = CROSSREF_TYPES.get(item.get("type", ""))
    if kind is None:
        return None
    titles = item.get("title") or []
    title = _clean_title(titles[0]) if titles else ""
    if len(title) < 8 or len(title) > 220:
        return None
    if not re.search(r"[A-Za-z]{3}", title):
        return None
    authors: list[dict[str, str]] = []
    for a in item.get("author") or []:
        fam = _clean_title(a.get("family", "") or "")
        giv = _clean_title(a.get("given", "") or "")
        if not fam and a.get("name"):
            continue  # organisations: skip, the renderer adds its own
        if not fam or len(fam) > 40 or len(giv) > 40:
            continue
        if not re.match(r"^[\w'\-\. ]+$", fam + giv, flags=re.UNICODE):
            continue
        authors.append({"given": giv, "family": fam})
    if not authors:
        return None
    editors: list[dict[str, str]] = []
    for e in item.get("editor") or []:
        if e.get("family") and e.get("given"):
            editors.append({"given": _clean_title(e["given"]), "family": _clean_title(e["family"])})
    issued = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = issued[0][0] if issued and issued[0] else None
    if not isinstance(year, int) or year < 1850 or year > 2027:
        return None
    container = item.get("container-title") or []
    container = _clean_title(container[0]) if container else ""
    short = item.get("short-container-title") or []
    short = _clean_title(short[0]) if short else ""
    page = item.get("page") or ""
    pages = None
    m = re.match(r"^([A-Za-z]?\d+)\s*[-–]\s*([A-Za-z]?\d+)$", page)
    if m:
        pages = {"from": m.group(1), "to": m.group(2)}
    elif re.match(r"^[A-Za-z]?\d{1,7}$", page):
        pages = {"from": page, "to": ""}
    rec: dict[str, Any] = {
        "source": "crossref",
        "type": kind,
        "authors": authors,
        "title": title,
        "year": year,
        "doi": item.get("DOI", "").lower() if item.get("DOI") else "",
    }
    if editors:
        rec["editors"] = editors
    if container:
        rec["container"] = container
    if short and short != container:
        rec["container_short"] = short
    if item.get("volume") and re.match(r"^[\w\-]{1,8}$", str(item["volume"])):
        rec["volume"] = str(item["volume"])
    if item.get("issue") and re.match(r"^[\w\-– ]{1,10}$", str(item["issue"])):
        rec["issue"] = str(item["issue"])
    if pages:
        rec["pages"] = pages
    if item.get("publisher"):
        rec["publisher"] = _clean_title(item["publisher"])
    if item.get("publisher-location"):
        rec["location"] = _clean_title(item["publisher-location"])
    if item.get("edition-number"):
        rec["edition"] = str(item["edition-number"])
    ev = item.get("event") or {}
    if ev.get("name"):
        rec["event"] = _clean_title(ev["name"])
        if ev.get("location"):
            rec["location"] = _clean_title(ev["location"])
    inst = item.get("institution") or []
    if inst and isinstance(inst, list) and inst[0].get("name"):
        rec["institution"] = _clean_title(inst[0]["name"])
    return rec


def fetch_crossref(n_requests: int = 60, force: bool = False) -> list[dict[str, Any]]:
    """Sample ``100 * n_requests`` works from CrossRef and normalise them."""
    out = CACHE / "crossref.jsonl"
    if out.exists() and not force:
        return [json.loads(line) for line in out.read_text().splitlines() if line]
    CACHE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(n_requests):
        url = f"https://api.crossref.org/works?sample=100&mailto={urllib.parse.quote(MAILTO)}"
        try:
            data = json.loads(_get(url))
        except RuntimeError as exc:
            print(f"crossref request {i} failed: {exc}")
            continue
        for item in data.get("message", {}).get("items", []):
            rec = _crossref_record(item)
            if rec and rec["doi"] not in seen:
                seen.add(rec["doi"])
                records.append(rec)
        print(f"crossref: {i + 1}/{n_requests} requests, {len(records)} records", flush=True)
        time.sleep(0.5)
    with out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


# --------------------------------------------------------------------------------- arXiv

ARXIV_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}


def _arxiv_record(rec: ET.Element) -> dict[str, Any] | None:
    meta = rec.find("oai:metadata/ax:arXiv", ARXIV_NS)
    if meta is None:
        return None
    ident = meta.findtext("ax:id", default="", namespaces=ARXIV_NS)
    title = _clean_title(meta.findtext("ax:title", default="", namespaces=ARXIV_NS))
    created = meta.findtext("ax:created", default="", namespaces=ARXIV_NS)
    if not ident or len(title) < 8 or len(title) > 220 or not created:
        return None
    authors = []
    for a in meta.findall("ax:authors/ax:author", ARXIV_NS):
        fam = _clean_title(a.findtext("ax:keyname", default="", namespaces=ARXIV_NS))
        giv = _clean_title(a.findtext("ax:forenames", default="", namespaces=ARXIV_NS))
        if fam and giv and len(fam) < 40 and len(giv) < 40:
            authors.append({"given": giv, "family": fam})
    if not authors:
        return None
    year = int(created[:4])
    out: dict[str, Any] = {
        "source": "arxiv",
        "type": "preprint",
        "authors": authors,
        "title": title,
        "year": year,
        "arxiv": ident,
        "month": int(created[5:7]),
    }
    cats = meta.findtext("ax:categories", default="", namespaces=ARXIV_NS).split()
    if cats:
        out["category"] = cats[0]
    doi = meta.findtext("ax:doi", default="", namespaces=ARXIV_NS)
    if doi:
        out["doi"] = doi.split()[0].lower()
    jref = meta.findtext("ax:journal-ref", default="", namespaces=ARXIV_NS)
    if jref:
        out["journal_ref"] = _clean_title(jref)
    return out


def fetch_arxiv(max_pages: int = 6, force: bool = False) -> list[dict[str, Any]]:
    """Harvest a few thousand arXiv records via OAI-PMH (one page ≈ 1000 records)."""
    out = CACHE / "arxiv.jsonl"
    if out.exists() and not force:
        return [json.loads(line) for line in out.read_text().splitlines() if line]
    CACHE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    # Spread over several dates/sets to diversify categories and identifier eras.
    windows = [
        ("2024-01-01", "2024-01-02", "cs"),
        ("2023-06-05", "2023-06-05", "physics"),
        ("2022-03-03", "2022-03-03", "math"),
        ("2021-09-14", "2021-09-14", "q-bio"),
        ("2020-11-10", "2020-11-10", "stat"),
        ("2019-05-20", "2019-05-20", "econ"),
        ("2018-02-12", "2018-02-12", "eess"),
    ]
    for start, until, subset in windows[:max_pages]:
        url = (
            "https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv"
            f"&set={subset}&from={start}&until={until}"
        )
        try:
            root = ET.fromstring(_get(url, timeout=120))
        except (RuntimeError, ET.ParseError) as exc:
            print(f"arxiv window {start} failed: {exc}")
            continue
        for rec in root.findall("oai:ListRecords/oai:record", ARXIV_NS):
            r = _arxiv_record(rec)
            if r:
                records.append(r)
        print(f"arxiv: {subset} {start}: {len(records)} records", flush=True)
        time.sleep(3.0)
    with out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


# ------------------------------------------------------------------------------ anystyle

ANYSTYLE_URL = "https://raw.githubusercontent.com/inukshuk/anystyle/main/res/parser/core.xml"


def fetch_anystyle_core(force: bool = False) -> Path:
    """Download anystyle's hand-labelled core corpus (BSD-2-Clause). Evaluation only."""
    out = CACHE / "anystyle_core.xml"
    if out.exists() and not force:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(ANYSTYLE_URL))
    return out


# -------------------------------------------------------------------------------- GROBID

GROBID_TREE = "https://api.github.com/repos/grobidOrg/grobid/git/trees/master?recursive=1"
GROBID_RAW = "https://raw.githubusercontent.com/grobidOrg/grobid/master/"
GROBID_DIR = "grobid-trainer/resources/dataset/citation/"


def fetch_grobid_citations(force: bool = False, limit: int = 400) -> Path:
    """Download GROBID's hand-labelled citation TEI corpus (Apache-2.0). Evaluation only.

    Files are concatenated into one ``grobid_citations.xml`` so a single cached artefact
    survives between runs. Only the full reference-list files are used (the small
    ``header-reference`` files hold a single self-citation each).
    """
    out = CACHE / "grobid_citations.xml"
    if out.exists() and not force:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    tree = json.loads(_get(GROBID_TREE))
    paths = sorted(
        t["path"]
        for t in tree.get("tree", [])
        if t["path"].startswith(GROBID_DIR)
        and (t["path"].endswith(".references.tei.xml") or t["path"].endswith(".references.xml"))
    )[:limit]
    parts = ["<grobid>"]
    for i, path in enumerate(paths):
        try:
            text = _get(GROBID_RAW + urllib.parse.quote(path), timeout=60).decode("utf-8", "replace")
        except RuntimeError as exc:
            print(f"grobid {path} failed: {exc}")
            continue
        text = re.sub(r"<\?xml[^>]*\?>", "", text)
        parts.append(f'<file path="{path}">{text}</file>')
        if (i + 1) % 25 == 0:
            print(f"grobid: {i + 1}/{len(paths)} files", flush=True)
    parts.append("</grobid>")
    out.write_text("\n".join(parts))
    return out


def load_metadata(seed: int = 0) -> list[dict[str, Any]]:
    """All structured records, shuffled deterministically."""
    recs = fetch_crossref() + fetch_arxiv()
    random.Random(seed).shuffle(recs)
    return recs


if __name__ == "__main__":
    cr = fetch_crossref()
    ax = fetch_arxiv()
    fetch_anystyle_core()
    fetch_grobid_citations()
    from collections import Counter

    print(len(cr), "crossref records", Counter(r["type"] for r in cr).most_common())
    print(len(ax), "arxiv records")
