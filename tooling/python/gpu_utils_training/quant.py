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


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """Exactly what runtime/weights.ts computes: float64 product, stored as float32."""
    return (q.astype(np.float64) * float(scale)).astype(np.float32)


def fake_quant(t: np.ndarray) -> np.ndarray:
    """Round-trip through int6 (the NumPy twin of qat.fake_quant; bit-identical to the runtime)."""
    q, scale = quantize(np.asarray(t, dtype=np.float32))
    return dequantize(q, scale)


LOOKUP = {ch: i - 32 for i, ch in enumerate(ALPHABET)}


def decode(encoded: str) -> np.ndarray:
    """Inverse of encode(): int codes in [-31, 31]."""
    return np.fromiter((LOOKUP[c] for c in encoded), dtype=np.int32, count=len(encoded))


def decode_weights(encoded: str, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Decode weights.txt back to named float32 tensors, exactly like runtime/weights.ts."""
    out: dict[str, np.ndarray] = {}
    for t in manifest["tensors"]:
        q = decode(encoded[t["offset"] : t["offset"] + t["length"]])
        out[t["name"]] = dequantize(q, t["scale"]).reshape(t["shape"])
    return out


def flat_weights(encoded: str, manifest: dict[str, Any]) -> np.ndarray:
    """The flat float32 buffer the WGSL kernels index with manifest offsets (decodeInt6 in TS)."""
    out = np.zeros(len(encoded), dtype=np.float32)
    for t in manifest["tensors"]:
        seg = encoded[t["offset"] : t["offset"] + t["length"]]
        out[t["offset"] : t["offset"] + t["length"]] = dequantize(decode(seg), t["scale"])
    return out


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
