"""NumPy Viterbi and BIO transition tables. Mirrors packages/runtime/src/decode.ts and
bio.ts; parity is enforced by packages/runtime/test/fixtures/viterbi.json on both sides.
"""

from __future__ import annotations

import numpy as np

FORBID = -1e9


def viterbi(emissions: np.ndarray, transitions: np.ndarray) -> list[int]:
    """Best path over ``[n, k]`` emissions with ``[k, k]`` transitions (``[from, to]``).

    Ties resolve to the lowest index, exactly like decode.ts (strict ``>`` comparisons).
    """
    em = np.asarray(emissions, dtype=np.float64)
    tr = np.asarray(transitions, dtype=np.float64)
    n = em.shape[0]
    if n == 0:
        return []
    k = em.shape[1]
    score = np.empty((n, k))
    back = np.zeros((n, k), dtype=np.int64)
    score[0] = em[0]
    for i in range(1, n):
        cand = score[i - 1][:, None] + tr  # [from, to]
        back[i] = np.argmax(cand, axis=0)  # first maximum, like the TS loop
        score[i] = cand[back[i], np.arange(k)] + em[i]
    path = [0] * n
    path[-1] = int(np.argmax(score[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = int(back[i, path[i]])
    return path


def bio_transitions(labels: list[str], forbid: float = FORBID) -> np.ndarray:
    """``[k, k]`` float32 table forbidding ``O -> I-X``, ``B-X -> I-Y`` and ``I-X -> I-Y`` (X != Y).

    Labels are ``"O"``, ``"B-X"`` or ``"I-X"``; anything else is treated as an unconstrained tag.
    """
    k = len(labels)
    t = np.zeros((k, k), dtype=np.float32)
    for to, lt in enumerate(labels):
        if not lt.startswith("I-"):
            continue
        for frm, lf in enumerate(labels):
            ok = (lf.startswith("B-") or lf.startswith("I-")) and lf[2:] == lt[2:]
            if not ok:
                t[frm, to] = forbid
    return t


def bio_start_mask(labels: list[str], forbid: float = FORBID) -> np.ndarray:
    """Additive ``[k]`` penalty for the first token: a sequence may not start with ``I-X``."""
    return np.array([forbid if lab.startswith("I-") else 0.0 for lab in labels], dtype=np.float32)
