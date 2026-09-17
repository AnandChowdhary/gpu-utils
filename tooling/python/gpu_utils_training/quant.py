"""int6 symmetric per-tensor quantization and the text encoding read by runtime/weights.ts.

Encoding: each weight is one character from ALPHABET (64 printable ASCII chars, no
quotes/backslashes), representing an integer in [-31, 31] (code 0 = -32 unused).
The manifest carries one float scale per tensor plus offsets into the flat string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ALPHABET = "".join(chr(c) for c in range(35, 127) if chr(c) not in "\\\"'`")[:64]
assert len(ALPHABET) == 64
LEVELS = 31


def quantize(t: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(t))) / LEVELS if t.size else 1.0
    scale = scale or 1.0
    q = np.clip(np.round(t / scale), -LEVELS, LEVELS).astype(np.int32)
    return q, scale


def fake_quant(t: np.ndarray) -> np.ndarray:
    """Round-trip through int6 (use with a straight-through estimator during QAT)."""
    q, scale = quantize(t)
    return q.astype(np.float32) * scale


def encode(q: np.ndarray) -> str:
    return "".join(ALPHABET[int(v) + 32] for v in q.ravel())


def export(tensors: dict[str, np.ndarray], out_dir: Path, manifest_extra: dict[str, Any]) -> dict[str, Any]:
    """Write weights.txt + manifest.json. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    entries: list[dict[str, Any]] = []
    offset = 0
    for name, t in tensors.items():
        q, scale = quantize(np.asarray(t, dtype=np.float32))
        s = encode(q)
        entries.append({"name": name, "offset": offset, "length": q.size, "shape": list(t.shape), "scale": scale})
        parts.append(s)
        offset += q.size
    (out_dir / "weights.txt").write_text("".join(parts))
    manifest = {"format": 1, **manifest_extra, "tensors": entries, "parameters": offset}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
