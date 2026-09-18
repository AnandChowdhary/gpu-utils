"""Real-world evaluation set derived from Loghub's 2k samples.

Loghub (LogPAI, https://github.com/logpai/loghub) ships, for 16 systems, 2,000 raw log
lines and a `*_structured.csv` whose columns are the hand-checked fields of each line. This
module turns those columns into gold BIO spans in *our* role vocabulary, mechanically:

1. every column of a system is searched for in the raw line, left to right, from a cursor
   (so repeated values such as Loghub's `Node`/`NodeRepeat` land on the right occurrence);
2. a column is mapped to one of our roles, merged into the previous span (`+`, used to join
   `Date` + `Time` into one TS), or mapped to `None` (a field with no counterpart in our
   vocabulary, e.g. Loghub's leading BGL/Thunderbird epoch or OpenStack's composite ADDR);
3. every gap the columns do not cover is filler. A gap of punctuation and whitespace is
   gold `O`; a gap that contains letters or digits is *unlabelled* (Loghub simply does not
   say what it is, e.g. the `sshd` between host and pid in OpenSSH) and becomes an **ignore
   region**: tokens inside it are dropped from token accuracy, and predictions that fall
   entirely inside it are dropped before span scoring;
4. a line whose columns cannot be found in order is dropped entirely.

The result is real, third-party-labelled data that the generator never produced. Nothing
here is committed: the samples are downloaded at evaluation time into
`training/data/cache/loghub/`. Loghub's terms make the datasets "freely available for
research or academic work" and ask for a citation (see THIRD_PARTY_NOTICES.md); they are
used for evaluation only and never for training.

    uv run python -m gpu_log.loghub [--limit 1000]   # coverage report + sample lines
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA_DIR / "cache" / "loghub"
RAW_URL = "https://raw.githubusercontent.com/logpai/loghub/master/{d}/{d}_2k.log{suffix}"

# Column -> role, in the order the columns appear in the raw line. "+ROLE" extends the
# previous span (Date + Time -> one TS). None marks a column we deliberately do not score.
Mapping = list[tuple[str, str | None]]

MAPPINGS: dict[str, Mapping] = {
    "Android": [("Date", "TS"), ("Time", "+TS"), ("Pid", "THREAD"), ("Tid", "THREAD"), ("Level", "LEVEL"), ("Component", "SOURCE"), ("Content", "MSG")],
    "Apache": [("Time", "TS"), ("Level", "LEVEL"), ("Content", "MSG")],
    # Loghub's leading BGL fields are a label, an epoch and a date that repeat the Time column.
    "BGL": [("Label", None), ("Timestamp", None), ("Date", None), ("Node", "HOST"), ("Time", "TS"), ("NodeRepeat", "HOST"), ("Type", None), ("Component", "SOURCE"), ("Level", "LEVEL"), ("Content", "MSG")],
    "HDFS": [("Date", "TS"), ("Time", "+TS"), ("Pid", "THREAD"), ("Level", "LEVEL"), ("Component", "SOURCE"), ("Content", "MSG")],
    "HPC": [("LogId", None), ("Node", "HOST"), ("Component", "SOURCE"), ("State", None), ("Time", "TS"), ("Flag", None), ("Content", "MSG")],
    "Hadoop": [("Date", "TS"), ("Time", "+TS"), ("Level", "LEVEL"), ("Process", "THREAD"), ("Component", "SOURCE"), ("Content", "MSG")],
    "HealthApp": [("Time", "TS"), ("Component", "SOURCE"), ("Pid", "THREAD"), ("Content", "MSG")],
    # Loghub's Linux "Level" column is really the hostname (`combo`).
    "Linux": [("Month", "TS"), ("Date", "+TS"), ("Time", "+TS"), ("Level", "HOST"), ("Component", "SOURCE"), ("PID", "THREAD"), ("Content", "MSG")],
    "Mac": [("Month", "TS"), ("Date", "+TS"), ("Time", "+TS"), ("User", "HOST"), ("Component", "SOURCE"), ("PID", "THREAD"), ("Address", None), ("Content", "MSG")],
    # OpenSSH's "Component" is the host; the process name (`sshd`) has no column and is unlabelled.
    "OpenSSH": [("Date", "TS"), ("Day", "+TS"), ("Time", "+TS"), ("Component", "HOST"), ("Pid", "THREAD"), ("Content", "MSG")],
    # OpenStack ADDR is `[<request id> <tenant> <user> - - -]`: a composite with no single role.
    "OpenStack": [("Logrecord", None), ("Date", "TS"), ("Time", "+TS"), ("Pid", "THREAD"), ("Level", "LEVEL"), ("Component", "SOURCE"), ("ADDR", None), ("Content", "MSG")],
    "Proxifier": [("Time", "TS"), ("Program", "SOURCE"), ("Content", "MSG")],
    "Spark": [("Date", "TS"), ("Time", "+TS"), ("Level", "LEVEL"), ("Component", "SOURCE"), ("Content", "MSG")],
    "Thunderbird": [("Label", None), ("Timestamp", None), ("Date", None), ("User", "HOST"), ("Month", "TS"), ("Day", "+TS"), ("Time", "+TS"), ("Location", "HOST"), ("Component", "SOURCE"), ("PID", "THREAD"), ("Content", "MSG")],
    "Windows": [("Date", "TS"), ("Time", "+TS"), ("Level", "LEVEL"), ("Component", "SOURCE"), ("Content", "MSG")],
    # Zookeeper's bracket is `[<thread>:<logger>@<source line>]`.
    "Zookeeper": [("Date", "TS"), ("Time", "+TS"), ("Level", "LEVEL"), ("Node", "THREAD"), ("Component", "SOURCE"), ("Id", None), ("Content", "MSG")],
}

SYSTEMS = list(MAPPINGS)
# Layouts the v2 synthetic generator imitates (formats.py). The rest are unseen shapes.
SEEN_LAYOUTS = {"HDFS", "Hadoop", "Spark", "Zookeeper", "OpenSSH", "Linux", "Mac", "Apache", "Android", "BGL", "HPC", "Thunderbird", "Windows", "Proxifier"}


@dataclass(frozen=True)
class RealLine:
    system: str
    text: str
    spans: tuple[tuple[int, int, str], ...]
    ignores: tuple[tuple[int, int], ...]


def fetch(system: str, suffix: str) -> Path | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{system}_2k.log{suffix}"
    if not path.exists():
        try:
            with urllib.request.urlopen(RAW_URL.format(d=system, suffix=suffix), timeout=60) as response:
                path.write_bytes(response.read())
        except Exception as exc:  # noqa: BLE001 - offline is a skip, not a failure
            print(f"  (skipping {system}: {exc})")
            return None
    return path


def align(line: str, row: dict[str, str], mapping: Mapping) -> RealLine | None:
    """Locate every column in the raw line and turn the mapping into spans + ignore regions."""
    spans: list[list[object]] = []
    covered: list[tuple[int, int]] = []
    unlabelled: list[tuple[int, int]] = []
    cursor = 0
    for column, role in mapping:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        start = line.find(value, cursor)
        if start < 0:
            return None
        end = start + len(value)
        cursor = end
        covered.append((start, end))
        if role is None:
            unlabelled.append((start, end))
            continue
        if role.startswith("+"):
            if not spans or spans[-1][2] != role[1:]:
                return None
            spans[-1][1] = end  # extend the previous span over the gap
        else:
            spans.append([start, end, role])
    if not spans:
        return None
    ignores = list(unlabelled)
    for (_, gap_start), (gap_end, _) in zip([(0, 0), *covered], [*covered, (len(line), len(line))], strict=True):
        gap = line[gap_start:gap_end]
        if any(c.isalnum() for c in gap):
            lead = len(gap) - len(gap.lstrip())
            trail = len(gap) - len(gap.rstrip())
            ignores.append((gap_start + lead, gap_end - trail))
    merged: list[list[int]] = []
    for start, end in sorted(ignores):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return RealLine(row["__system__"], line, tuple((int(s), int(e), str(r)) for s, e, r in spans), tuple((a, b) for a, b in merged))


def load(limit: int = 1000, systems: list[str] | None = None) -> tuple[list[RealLine], dict[str, tuple[int, int]]]:
    """Aligned real lines plus {system: (kept, seen)}. Exact duplicate lines are dropped."""
    out: list[RealLine] = []
    coverage: dict[str, tuple[int, int]] = {}
    for system in systems or SYSTEMS:
        raw_path, csv_path = fetch(system, ""), fetch(system, "_structured.csv")
        if raw_path is None or csv_path is None:
            continue
        lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8", errors="replace"))))
        kept, seen_texts = 0, set()
        for line, row in zip(lines, rows, strict=False):
            if kept >= limit:
                break
            text = line.rstrip("\n")
            if not text.strip() or text in seen_texts:
                continue
            seen_texts.add(text)
            row["__system__"] = system
            aligned = align(text, row, MAPPINGS[system])
            if aligned is not None:
                out.append(aligned)
                kept += 1
        coverage[system] = (kept, len(seen_texts))
    return out, coverage


def to_markup(rl: RealLine) -> str:
    """Debug rendering: ⟦ROLE|…⟧ for gold spans, ⟨…⟩ for unlabelled (ignored) regions."""
    marks = sorted([(s, e, r) for s, e, r in rl.spans] + [(s, e, None) for s, e in rl.ignores])
    out, pos = [], 0
    for s, e, role in marks:
        out.append(rl.text[pos:s])
        out.append(f"⟦{role}|{rl.text[s:e]}⟧" if role else f"⟨{rl.text[s:e]}⟩")
        pos = e
    out.append(rl.text[pos:])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and inspect the Loghub-derived evaluation set")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    lines, coverage = load(args.limit)
    print(f"{len(lines)} aligned lines from {len(coverage)} systems (limit {args.limit}/system, duplicates dropped)\n")
    for system, (kept, seen) in coverage.items():
        tag = "generator imitates this layout" if system in SEEN_LAYOUTS else "unseen layout"
        print(f"  {system:12s} {kept:5d}/{seen:5d} aligned  ({tag})")
        for rl in [x for x in lines if x.system == system][: args.samples]:
            print(f"      {to_markup(rl)[:200]}")


if __name__ == "__main__":
    main()
