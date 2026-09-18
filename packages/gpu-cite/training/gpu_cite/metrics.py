"""Span-level scoring for gpu-cite.

The package-specific part is span *normalisation*: edge punctuation and cue words
(``vol.``, ``pp.``, ``(Eds.)``) are trimmed from gold and predicted spans so that different
tagging conventions (ours, anystyle's, GROBID's) agree on the same field, and AUTHOR /
EDITOR lists are collapsed to their union span. Precision / recall / F1 come from
``gpu_utils_training.metrics.span_prf``; record-level exact match, type accuracy and
author-entity F1 are accumulated here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gpu_utils_training.features import Token
from gpu_utils_training.metrics import Span, format_prf_table, span_prf

from .labels import ROLES, spans_from_tags

TRIM = " \t\n.,;:()[]\"'“”‘’«»‚„"
CUE_WORDS = {"vol", "vol.", "volume", "no", "no.", "number", "issue", "pp", "pp.", "p", "p.", "pages", "page",
             "ed", "ed.", "eds", "eds.", "edn", "edn.", "edition", "in", "in:", "doi:", "doi", "arxiv:", "retrieved",
             "accessed", "accessed:", "available", "at:", "from", "editors", "editors.", "editor", "(eds.)", "(ed.)",
             "(eds)", "(ed)", "hrsg.", "(hrsg.)", "edited", "by", "https://doi.org/", "http://dx.doi.org/", "s."}
TRAILING_CUES = {"(eds", "(ed", "eds", "ed", "editors", "editor", "hrsg", "(hrsg", "et al", "et al.", "(eds.", "(ed."}
LIST_ROLES = {"AUTHOR", "EDITOR"}


def _utf16(text: str) -> str:
    """Return a string whose Python indices equal UTF-16 offsets (astral chars → 2 units)."""
    if all(ord(c) <= 0xFFFF for c in text):
        return text
    out = []
    for c in text:
        out.append(c if ord(c) <= 0xFFFF else "��")
    return "".join(out)


def normalize_span(text16: str, start: int, end: int) -> tuple[int, int]:
    """Trim edge punctuation/whitespace and leading cue words so different tag conventions agree."""
    while start < end and text16[start] in TRIM:
        start += 1
    while end > start and text16[end - 1] in TRIM:
        end -= 1
    cues = sorted(CUE_WORDS | TRAILING_CUES, key=len, reverse=True)
    changed = True
    while changed and start < end:
        changed = False
        low = text16[start:end].lower()
        for cue in cues:
            if cue in CUE_WORDS and low.startswith(cue) and (len(low) == len(cue) or not low[len(cue)].isalnum()):
                start += len(cue)
                while start < end and text16[start] in TRIM:
                    start += 1
                changed = True
                break
            if cue in TRAILING_CUES and low.endswith(cue) and (len(low) == len(cue) or not low[-len(cue) - 1].isalnum()):
                end -= len(cue)
                while end > start and text16[end - 1] in TRIM:
                    end -= 1
                changed = True
                break
    return start, end


def field_spans(text: str, tags: list[int], tokens: list[Token]) -> dict[str, set[tuple[int, int]]]:
    """Role → set of normalized (start, end). List roles are collapsed to their union span."""
    t16 = _utf16(text)
    raw = spans_from_tags(tags, tokens)
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for role, s, e in raw:
        if role in LIST_ROLES:
            continue
        ns = normalize_span(t16, s, e)
        if ns[0] < ns[1]:
            out[role].add(ns)
    for role in LIST_ROLES:
        ents = [(s, e) for r, s, e in raw if r == role]
        if ents:
            ns = normalize_span(t16, min(s for s, _ in ents), max(e for _, e in ents))
            if ns[0] < ns[1]:
                out[role].add(ns)
    return out


def entity_spans(text: str, tags: list[int], tokens: list[Token], role: str) -> list[tuple[int, int]]:
    t16 = _utf16(text)
    return [normalize_span(t16, s, e) for r, s, e in spans_from_tags(tags, tokens) if r == role]


class SpanScorer:
    """Accumulates records; ``summary()`` is the metrics dict used by train, evaluate and MODEL_CARD.md."""

    def __init__(self) -> None:
        self.gold: list[list[Span]] = []
        self.pred: list[list[Span]] = []
        self.gold_authors: list[list[Span]] = []
        self.pred_authors: list[list[Span]] = []
        self.records = 0
        self.exact = 0
        self.exact_fields = 0
        self.type_correct = 0
        self.type_total = 0

    def add(
        self,
        gold: dict[str, set[tuple[int, int]]],
        pred: dict[str, set[tuple[int, int]]],
        gold_type: int | None = None,
        pred_type: int | None = None,
        gold_authors: list[tuple[int, int]] | None = None,
        pred_authors: list[tuple[int, int]] | None = None,
        roles: list[str] | None = None,
        extra_ok: bool = True,
    ) -> None:
        roles = roles or ROLES
        self.records += 1
        self.gold.append([(r, s, e) for r in roles for s, e in sorted(gold.get(r, ()))])
        self.pred.append([(r, s, e) for r in roles for s, e in sorted(pred.get(r, ()))])
        all_ok = extra_ok and all(gold.get(r, set()) == pred.get(r, set()) for r in roles)
        self.exact_fields += all_ok
        if gold_type is not None:
            self.type_total += 1
            self.type_correct += gold_type == pred_type
            all_ok = all_ok and gold_type == pred_type
        self.exact += all_ok
        if gold_authors is not None and pred_authors is not None:
            self.gold_authors.append([("AUTHOR", s, e) for s, e in gold_authors])
            self.pred_authors.append([("AUTHOR", s, e) for s, e in pred_authors])

    def summary(self, roles: list[str] | None = None) -> dict[str, Any]:
        roles = roles or ROLES
        prf = span_prf(self.pred, self.gold)
        per, micro, macro = prf["per_label"], prf["micro"], prf["macro"]
        assert isinstance(per, dict) and isinstance(micro, dict) and isinstance(macro, dict)
        out: dict[str, Any] = {
            "records": self.records,
            "fields": {role: {k: round(v, 4) for k, v in per[role].items()} for role in roles if role in per},
            "micro": {k: round(v, 4) for k, v in micro.items()},
            "micro_f1": round(micro["f1"], 4),
            "macro_f1": round(macro["f1"], 4),
            "exact_fields": round(self.exact_fields / max(1, self.records), 4),
            "exact_match": round(self.exact / max(1, self.records), 4),
        }
        if self.type_total:
            out["type_accuracy"] = round(self.type_correct / self.type_total, 4)
        if self.gold_authors:
            authors = span_prf(self.pred_authors, self.gold_authors)["micro"]
            assert isinstance(authors, dict)
            out["author_entity_f1"] = round(authors["f1"], 4)
        return out


def format_table(summary: dict[str, Any]) -> str:
    """Markdown P/R/F1 table in role order (``gpu_utils_training.metrics.format_prf_table``)."""
    return format_prf_table({"per_label": summary["fields"], "micro": summary["micro"]})
