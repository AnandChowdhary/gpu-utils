"""Evaluate the promoted checkpoint with the shipped decoder on:

  heldout     generated set (data/cache/heldout.json, seed 999, never trained on)
  unfamiliar  hand-written set in styles the generator does not produce (data/unfamiliar.json)
  external    email_reply_parser fixtures (MIT) and talon fixtures (Apache 2.0), downloaded
              by `python -m gpu_email.data` into data/cache (reply / signature expectations
              in data/external_expected.json)

    uv run python -m gpu_email.evaluate [--markdown]

Reports line-kind accuracy (non-blank lines), reply exact match, and per-field contact
precision/recall/F1. Numbers use the int6-dequantized weights, i.e. what the package ships.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gpu_email.data import CACHE_DIR, DATA_DIR, reply_from_lines, Line, KIND
from gpu_email.decode import decode
from gpu_email.export import dequantized_model, load_model
from gpu_email.features import FIELDS, LINE_KINDS, featurize_tokens, line_infos
from gpu_email.model import EmailTagger
from gpu_utils_training.features import tokenize


def run(model: EmailTagger, text: str) -> dict:
    tokens = tokenize(text)
    lines = line_infos(tokens)
    if not tokens:
        return {"line_kinds": [], "segments": [], "reply": "", "contact": None}
    rows = featurize_tokens(tokens)
    ids = torch.tensor(rows, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        kind, bio = model(ids, torch.ones(1, len(rows)))
    logits = torch.cat([kind, bio], dim=-1)[0].numpy()
    return decode(tokens, lines, logits, text)


def load_unfamiliar() -> list[dict]:
    raw = json.loads((DATA_DIR / "unfamiliar.json").read_text())
    out = []
    for case in raw:
        lines = [Line(t, KIND[k] if k else KIND["reply"]) for k, t in case["lines"]]
        kinds = [(-1 if t.strip(" \t") == "" else KIND[k]) for k, t in case["lines"]]
        text = case.get("newline", "\n").join(t for _, t in case["lines"])
        if case.get("trailing_newline", True):
            text += case.get("newline", "\n")
        out.append({"name": case["name"], "text": text, "line_kinds": kinds, "reply": reply_from_lines(lines), "contact": case.get("contact") or {}})
    return out


def load_heldout(limit: int) -> list[dict]:
    p = CACHE_DIR / "heldout.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())[:limit]


def eml_body(raw: bytes) -> str:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    body = msg.get_body(preferencelist=("plain",))
    if body is None:
        return raw.decode("utf-8", errors="replace")
    return body.get_content()


def load_external() -> list[dict]:
    exp_path = DATA_DIR / "external_expected.json"
    if not exp_path.exists():
        return []
    expected = json.loads(exp_path.read_text())
    out = []
    for name, spec in expected.items():
        p = CACHE_DIR / spec["file"]
        if not p.exists():
            continue
        raw = p.read_bytes()
        text = eml_body(raw) if p.suffix == ".eml" else raw.decode("utf-8", errors="replace")
        text = text.replace("#sig#", "")  # talon's inline signature markers
        case = {"name": name, "text": text, "source": spec["source"]}
        if "reply" in spec:
            case["reply"] = spec["reply"]
        if "reply_line_ranges" in spec:
            rows = text.split("\n")
            keep = {i for a, b in spec["reply_line_ranges"] for i in range(a, min(b, len(rows) - 1) + 1)}
            case["reply"] = reply_from_lines([Line(t, KIND["reply"] if i in keep else KIND["quote"]) for i, t in enumerate(rows)])
        if "signature_file" in spec:
            case["signature"] = (CACHE_DIR / spec["signature_file"]).read_text().replace("#sig#", "")
        out.append(case)
    return out


class Metrics:
    def __init__(self) -> None:
        self.line_ok = 0
        self.line_total = 0
        self.per_kind = Counter()
        self.per_kind_ok = Counter()
        self.confusion = Counter()
        self.reply_ok = 0
        self.reply_total = 0
        self.sig_ok = 0
        self.sig_total = 0
        self.field = {f.lower(): Counter() for f in FIELDS}
        self.failures: list[str] = []

    def add_lines(self, gold: list[int], pred: list[int], name: str) -> None:
        n = min(len(gold), len(pred))
        for g, p in zip(gold[:n], pred[:n]):
            if g < 0:
                continue
            self.line_total += 1
            self.per_kind[LINE_KINDS[g]] += 1
            if g == p:
                self.line_ok += 1
                self.per_kind_ok[LINE_KINDS[g]] += 1
            else:
                self.confusion[(LINE_KINDS[g], LINE_KINDS[p] if p >= 0 else "blank")] += 1

    def add_reply(self, gold: str, pred: str, name: str) -> None:
        self.reply_total += 1
        if gold.strip() == pred.strip():
            self.reply_ok += 1
        else:
            self.failures.append(f"[reply] {name}: expected {gold[:60]!r} got {pred[:60]!r}")

    def add_contact(self, gold: dict, pred: dict | None, name: str) -> None:
        pred = pred or {}
        for f in FIELDS:
            key = f.lower()
            g = gold.get(key)
            p = pred.get(key)
            c = self.field[key]
            if key in ("phone", "email", "url"):
                gs, ps = set(g or []), set(p or [])
                c["tp"] += len(gs & ps)
                c["fp"] += len(ps - gs)
                c["fn"] += len(gs - ps)
                if ps - gs or gs - ps:
                    self.failures.append(f"[{key}] {name}: expected {sorted(gs)} got {sorted(ps)}")
            else:
                if g and p:
                    if g.strip() == p.strip():
                        c["tp"] += 1
                    else:
                        c["fp"] += 1
                        c["fn"] += 1
                        self.failures.append(f"[{key}] {name}: expected {g!r} got {p!r}")
                elif p:
                    c["fp"] += 1
                    self.failures.append(f"[{key}] {name}: expected none got {p!r}")
                elif g:
                    c["fn"] += 1
                    self.failures.append(f"[{key}] {name}: expected {g!r} got none")

    def rows(self) -> list[tuple[str, str]]:
        rows = []
        if self.line_total:
            rows.append(("line-kind accuracy", f"{self.line_ok / self.line_total:.4f} ({self.line_ok}/{self.line_total})"))
            for k in LINE_KINDS:
                if self.per_kind[k]:
                    rows.append((f"  {k}", f"{self.per_kind_ok[k] / self.per_kind[k]:.3f} (n={self.per_kind[k]})"))
        if self.reply_total:
            rows.append(("reply exact match", f"{self.reply_ok / self.reply_total:.4f} ({self.reply_ok}/{self.reply_total})"))
        if self.sig_total:
            rows.append(("signature exact match", f"{self.sig_ok / self.sig_total:.4f} ({self.sig_ok}/{self.sig_total})"))
        for key, c in self.field.items():
            if c["tp"] + c["fp"] + c["fn"] == 0:
                continue
            p = c["tp"] / max(1, c["tp"] + c["fp"])
            r = c["tp"] / max(1, c["tp"] + c["fn"])
            f1 = 2 * p * r / max(1e-9, p + r)
            rows.append((f"contact {key} F1", f"{f1:.3f} (P {p:.3f} R {r:.3f}, n={c['tp'] + c['fn']})"))
        return rows


def evaluate_set(model: EmailTagger, cases: list[dict]) -> Metrics:
    m = Metrics()
    for i, case in enumerate(cases):
        name = case.get("name", str(i))
        pred = run(model, case["text"])
        if "line_kinds" in case:
            m.add_lines(case["line_kinds"], pred["line_kinds"], name)
        if "reply" in case:
            m.add_reply(case["reply"], pred["reply"], name)
        if "contact" in case:
            m.add_contact(case["contact"], pred["contact"], name)
        if "signature" in case:
            m.sig_total += 1
            sig_text = ""
            for kind, s, e in pred["segments"]:
                if kind == "signature":
                    sig_text = case["text"].encode("utf-16-le")[s * 2 : e * 2].decode("utf-16-le", errors="ignore")
                    break
            if sig_text.strip() == case["signature"].strip():
                m.sig_ok += 1
            else:
                m.failures.append(f"[signature] {name}: expected {case['signature'][:60]!r} got {sig_text[:60]!r}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--heldout", type=int, default=3000)
    ap.add_argument("--failures", type=int, default=25)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, _ = load_model()
    model = dequantized_model(model)
    sets = {"held-out (generated)": load_heldout(args.heldout), "unfamiliar (hand-written)": load_unfamiliar(), "external (talon + email_reply_parser)": load_external()}
    for name, cases in sets.items():
        if not cases:
            print(f"\n## {name}: no cases found", file=sys.stderr)
            continue
        m = evaluate_set(model, cases)
        print(f"\n## {name} ({len(cases)} emails)")
        if args.markdown:
            print("| Metric | Value |\n|---|---|")
            for k, v in m.rows():
                print(f"| {k} | {v} |")
        else:
            for k, v in m.rows():
                print(f"{k:32s} {v}")
        if m.confusion:
            print("top confusions (gold → pred):", ", ".join(f"{g}→{p}: {n}" for (g, p), n in m.confusion.most_common(8)))
        for f in m.failures[: args.failures]:
            print("  ", f)


if __name__ == "__main__":
    main()
