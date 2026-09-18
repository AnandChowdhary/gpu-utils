"""Span-level metrics: per-field exact-span P/R/F1 and full-record exact match."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gpu_utils_training.features import Token

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
    """Accumulates per-role TP/FP/FN over records plus exact-match counts."""

    def __init__(self) -> None:
        self.tp: dict[str, int] = defaultdict(int)
        self.fp: dict[str, int] = defaultdict(int)
        self.fn: dict[str, int] = defaultdict(int)
        self.records = 0
        self.exact = 0
        self.exact_fields = 0
        self.type_correct = 0
        self.type_total = 0
        self.author_tp = 0
        self.author_fp = 0
        self.author_fn = 0

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
        all_ok = extra_ok
        for role in roles:
            g, p = gold.get(role, set()), pred.get(role, set())
            self.tp[role] += len(g & p)
            self.fp[role] += len(p - g)
            self.fn[role] += len(g - p)
            if g != p:
                all_ok = False
        self.exact_fields += all_ok
        if gold_type is not None:
            self.type_total += 1
            self.type_correct += gold_type == pred_type
            all_ok = all_ok and gold_type == pred_type
        self.exact += all_ok
        if gold_authors is not None and pred_authors is not None:
            g, p = set(gold_authors), set(pred_authors)
            self.author_tp += len(g & p)
            self.author_fp += len(p - g)
            self.author_fn += len(g - p)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    def summary(self, roles: list[str] | None = None) -> dict[str, Any]:
        roles = roles or ROLES
        per: dict[str, dict[str, float]] = {}
        ttp = tfp = tfn = 0
        for role in roles:
            tp, fp, fn = self.tp[role], self.fp[role], self.fn[role]
            if tp + fp + fn == 0:
                continue
            p, r, f = self._prf(tp, fp, fn)
            per[role] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4), "support": tp + fn}
            ttp, tfp, tfn = ttp + tp, tfp + fp, tfn + fn
        mp, mr, mf = self._prf(ttp, tfp, tfn)
        out: dict[str, Any] = {
            "records": self.records,
            "fields": per,
            "micro_f1": round(mf, 4),
            "macro_f1": round(sum(v["f1"] for v in per.values()) / max(1, len(per)), 4),
            "exact_fields": round(self.exact_fields / max(1, self.records), 4),
            "exact_match": round(self.exact / max(1, self.records), 4),
        }
        if self.type_total:
            out["type_accuracy"] = round(self.type_correct / self.type_total, 4)
        if self.author_tp + self.author_fp + self.author_fn:
            p, r, f = self._prf(self.author_tp, self.author_fp, self.author_fn)
            out["author_entity_f1"] = round(f, 4)
        return out


def format_table(summary: dict[str, Any]) -> str:
    lines = ["| Field | P | R | F1 | Support |", "|---|---|---|---|---|"]
    for role, v in summary["fields"].items():
        lines.append(f"| {role} | {v['p']:.3f} | {v['r']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines.append(f"| **micro** | | | **{summary['micro_f1']:.3f}** | |")
    return "\n".join(lines)
