"""Python mirror of src/decode.ts, used by evaluate.py so the reported metrics come from
the same rules + Viterbi + contact extraction the package ships. Keep in sync."""

from __future__ import annotations

import re

import numpy as np

from gpu_email.features import BIO_LABELS, EMAIL_RE, LINE_KINDS, LineInfo, URL_RE
from gpu_utils_training.features import CLASS_NEWLINE, CLASS_SPACE, Token

REPLY, ATTRIBUTION, QUOTE, SIGNATURE, DISCLAIMER, FORWARD, GREETING, CLOSING = range(8)
NEG = -1e9
TRANSITIONS = np.array(
    [
        [0.0, 0.0, -0.5, -0.5, -0.5, 0.0, -2.5, 0.0],
        [-3.0, 0.0, 0.0, -4.0, -3.0, -2.0, -3.0, -4.0],
        [-1.5, -0.5, 0.0, -2.0, -1.0, -1.0, -2.0, -2.0],
        [-3.0, -0.5, -1.5, 0.0, -0.5, -0.5, -4.0, -3.0],
        [-3.0, -0.5, -1.0, -1.0, 0.0, -0.5, -3.0, -3.0],
        [-3.0, -3.0, 0.0, -3.0, -3.0, 0.0, -3.0, -3.0],
        [0.0, -1.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0],
        [-1.5, -0.5, -1.0, 0.0, -0.5, -0.5, -3.0, -0.5],
    ],
    dtype=np.float64,
)


def viterbi(emissions: np.ndarray, transitions: np.ndarray) -> list[int]:
    """Same tie-breaking as runtime/decode.ts (first best on ties)."""
    n, k = emissions.shape
    if n == 0:
        return []
    score = np.zeros((n, k))
    back = np.zeros((n, k), dtype=np.int64)
    score[0] = emissions[0]
    for i in range(1, n):
        cand = score[i - 1][:, None] + transitions  # [from, to]
        back[i] = np.argmax(cand, axis=0)
        score[i] = cand[back[i], np.arange(k)] + emissions[i]
    path = [0] * n
    path[-1] = int(np.argmax(score[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = int(back[i, path[i]])
    return path


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def decode_line_kinds(tokens: list[Token], lines: list[LineInfo], logits: np.ndarray) -> list[int]:
    K = len(LINE_KINDS)
    lp = log_softmax(logits[:, :K].astype(np.float64)) if len(tokens) else np.zeros((0, K))
    kinds = [-1] * len(lines)
    active = [li for li, ln in enumerate(lines) if not ln.blank]
    if not active:
        return kinds
    em = np.zeros((len(active), K))
    delim_seen = False
    for a, li in enumerate(active):
        line = lines[li]
        idx = [i for i in range(line.start, line.end) if tokens[i].cls not in (CLASS_SPACE, CLASS_NEWLINE)]
        if not idx:
            idx = list(range(line.start, line.end))
        em[a] = lp[idx].mean(axis=0)
        if line.quote_prefixed:
            em[a] = NEG
            em[a, QUOTE] = 0
        elif line.delimiter:
            em[a] = NEG
            em[a, SIGNATURE] = 0
            delim_seen = True
        elif delim_seen:
            em[a, [REPLY, GREETING, CLOSING]] = NEG
    path = viterbi(em, TRANSITIONS)
    for a, li in enumerate(active):
        kinds[li] = path[a]
    return kinds


def build_segments(lines: list[LineInfo], kinds: list[int]) -> list[tuple[str, int, int]]:
    segs: list[list] = []
    for li, k in enumerate(kinds):
        if k < 0:
            continue
        ln = lines[li]
        if segs and segs[-1][0] == k:
            segs[-1][2] = ln.char_end
        else:
            segs.append([k, ln.char_start, ln.char_end])
    return [(LINE_KINDS[k], s, e) for k, s, e in segs]


def build_reply(lines: list[LineInfo], kinds: list[int]) -> str:
    out: list[str] = []
    for li, k in enumerate(kinds):
        if k in (REPLY, GREETING, CLOSING):
            out.append(lines[li].text.rstrip())
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def _u16(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _u16_slice(text: str, start: int, end: int) -> str:
    return text.encode("utf-16-le")[start * 2 : end * 2].decode("utf-16-le", errors="ignore")


def _bio_transitions() -> np.ndarray:
    B = len(BIO_LABELS)
    t = np.zeros((B, B))
    for f in range(B):
        for to in range(B):
            if BIO_LABELS[to].startswith("I-"):
                ok = BIO_LABELS[f] != "O" and BIO_LABELS[f][2:] == BIO_LABELS[to][2:]
                t[f, to] = 0 if ok else NEG
    return t


BIO_T = _bio_transitions()


def extract_contact(tokens: list[Token], logits: np.ndarray, text: str, span: tuple[int, int]) -> dict | None:
    K = len(LINE_KINDS)
    idx = [i for i, t in enumerate(tokens) if not (t.end <= span[0] or t.start >= span[1])]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    em = logits[first : last + 1, K:].astype(np.float64)
    path = viterbi(em, BIO_T)
    spans: list[list] = []
    for i, p in enumerate(path):
        label = BIO_LABELS[p]
        t = tokens[first + i]
        if label.startswith("B-"):
            spans.append([label[2:], t.start, t.end])
        elif label.startswith("I-") and spans:
            s = spans[-1]
            if s[0] == label[2:] and s[2] >= t.start:
                s[2] = t.end
    seg = _u16_slice(text, span[0], span[1])
    exact: list[list] = []
    for m in EMAIL_RE.finditer(seg):
        exact.append(["EMAIL", span[0] + _u16(seg[: m.start()]), span[0] + _u16(seg[: m.end()])])
    for m in URL_RE.finditer(seg):
        s = m.group(0)
        while s and s[-1] in ".,;:!?":
            s = s[:-1]
        start = span[0] + _u16(seg[: m.start()])
        end = start + _u16(s)
        if any(start < e[2] and end > e[1] for e in exact):
            continue
        exact.append(["URL", start, end])
    kept = [s for s in spans if s[0] not in ("EMAIL", "URL") and not any(s[1] < e[2] and s[2] > e[1] for e in exact)]
    allspans = sorted(kept + exact, key=lambda s: s[1])
    c: dict = {}
    any_field = False
    for f, s, e in allspans:
        value = _u16_slice(text, s, e).strip()
        if not value:
            continue
        if f == "PHONE":
            if len(re.findall(r"\d", value)) < 5:
                continue
            c.setdefault("phone", []).append(value)
        elif f == "EMAIL":
            c.setdefault("email", []).append(value)
        elif f == "URL":
            c.setdefault("url", []).append(value)
        elif f == "ADDRESS":
            c["address"] = (c["address"] + ", " + value) if "address" in c else value
        elif f in ("NAME", "TITLE", "COMPANY"):
            c.setdefault(f.lower(), value)
        else:
            continue
        any_field = True
    for key in ("phone", "email", "url"):
        if key in c:
            c[key] = list(dict.fromkeys(c[key]))
    return c if any_field else None


def author_signature(lines: list[LineInfo], kinds: list[int], segments) -> tuple[int, int] | None:
    """The author's own signature: the first signature segment after the first new-content
    line (or the first signature at all when there is no new content)."""
    first_reply = -1
    for li, k in enumerate(kinds):
        if k in (REPLY, GREETING, CLOSING):
            first_reply = lines[li].char_start
            break
    for kind, s, e in segments:
        if kind == "signature" and s >= first_reply:
            return (s, e)
    return None


def decode(tokens: list[Token], lines: list[LineInfo], logits: np.ndarray, text: str) -> dict:
    kinds = decode_line_kinds(tokens, lines, logits)
    segments = build_segments(lines, kinds)
    reply = build_reply(lines, kinds)
    sig = author_signature(lines, kinds, segments)
    contact = extract_contact(tokens, logits, text, sig) if sig else None
    return {"line_kinds": kinds, "segments": segments, "reply": reply, "contact": contact}
