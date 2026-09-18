"""Python mirror of packages/runtime/src/gpu.ts: runs the canonical WGSL kernels through
WgslRunner with exactly the buffers, uniform block and dispatch list the TypeScript
runtime uses. Used by the parity tests (TS CPU == Python == WGSL) and available to
packages that want to check a trained model on lavapipe before shipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .batch import pack_rows
from .wgsl import Uniform, WgslRunner

RUNTIME_WGSL = Path(__file__).resolve().parents[3] / "packages" / "runtime" / "src" / "wgsl"
SCAN_TAGGER_MAX_LAYERS = 4
CONV_TAGGER_MAX_BLOCKS = 8
GRID_X = 32768


def scan_tensor_names(m: dict[str, Any]) -> list[str]:
    names = ["embed"]
    for layer in range(m.get("scanLayers", 1)):
        for d in ("f", "b"):
            for p in ("wa", "ba", "wu", "bu"):
                names.append(f"scan{layer}.{d}.{p}")
        names += [f"conv{layer}.w", f"conv{layer}.b"]
    names += ["head.w", "head.b", "tags.w", "tags.b"]
    if m.get("pooled", 0) > 0:
        names += ["pool.w1", "pool.b1", "pool.w2", "pool.b2"]
    return names


def conv_tensor_names(m: dict[str, Any]) -> list[str]:
    names = ["embed", "proj.w", "proj.b"]
    for i in range(len(m.get("dilations", []))):
        names += [f"block{i}.w1", f"block{i}.b1", f"block{i}.w2", f"block{i}.b2"]
    names += ["head.w", "head.b", "tags.w", "tags.b"]
    if m.get("pooled", 0) > 0:
        names += ["pool.w1", "pool.b1", "pool.w2", "pool.b2"]
    return names


def scan_tagger_entries(layers: int) -> list[str]:
    if layers > SCAN_TAGGER_MAX_LAYERS:
        raise ValueError(f"scan_tagger.wgsl supports at most {SCAN_TAGGER_MAX_LAYERS} scan layers")
    return ["embed", *[e for l in range(layers) for e in (f"gates{l}", f"scan{l}", f"conv{l}")], "pool", "head", "pooled"]


def conv_tagger_entries(blocks: int) -> list[str]:
    if blocks > CONV_TAGGER_MAX_BLOCKS:
        raise ValueError(f"conv_tagger.wgsl supports at most {CONV_TAGGER_MAX_BLOCKS} blocks")
    return ["embed", *[f"block{i}" for i in range(blocks)], "head", "pool"]


def tensor_offsets(manifest: dict[str, Any], names: list[str]) -> np.ndarray:
    by_name = {t["name"]: t["offset"] for t in manifest["tensors"]}
    return np.array([by_name[n] for n in names], dtype=np.uint32)


def tagger_params(**p: int) -> np.ndarray:
    out = np.zeros(16, dtype=np.uint32)
    keys = ["batch", "max_tokens", "slots", "padding", "embed", "hidden", "head", "tags", "pooled", "layers", "taps"]
    out[: len(keys)] = [p[k] for k in keys]
    return out


def grid(n: int) -> tuple[int, int]:
    return (min(max(n, 1), GRID_X), max(1, -(-n // GRID_X)))


def shader_source(family: str) -> str:
    return (RUNTIME_WGSL / f"{family}_tagger.wgsl").read_text()


def make_runner(manifest: dict[str, Any], device: Any = None) -> WgslRunner:
    scan = manifest["family"] == "scan"
    entries = scan_tagger_entries(manifest.get("scanLayers", 1)) if scan else conv_tagger_entries(len(manifest["dilations"]))
    return WgslRunner(shader_source(manifest["family"]), entries, device)


def run_tagger(
    runner: WgslRunner,
    manifest: dict[str, Any],
    weights: np.ndarray,
    batch: list[list[list[int]]],
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Run a batch through the canonical kernel. Returns per-sequence ``[n, tags]`` logits
    and per-sequence ``[pooled]`` logits (or None)."""
    m = manifest
    scan = m["family"] == "scan"
    n = len(batch)
    rows, lengths = pack_rows(batch, m["slots"], m["paddingId"])
    max_tokens = rows.shape[1]
    layers = m.get("scanLayers", 1) if scan else len(m["dilations"])
    names = scan_tensor_names(m) if scan else conv_tensor_names(m)
    # Head entries always occupy 8 table slots so conv_tagger.wgsl can find the dilations.
    pad = [] if m["pooled"] > 0 else [0, 0, 0, 0]
    table = np.concatenate([tensor_offsets(m, names), np.asarray(pad, dtype=np.uint32), np.asarray(m["dilations"] if not scan else [], dtype=np.uint32)])
    params = tagger_params(
        batch=n,
        max_tokens=max_tokens,
        slots=m["slots"],
        padding=m["paddingId"],
        embed=m.get("embed", m["hidden"]),
        hidden=m["hidden"],
        head=m["head"],
        tags=m["tags"],
        pooled=m["pooled"],
        layers=layers,
        taps=m.get("convTaps", 5),
    )
    width = 2 * m["hidden"] if scan else m["hidden"]
    scratch_size = 4 * n * max_tokens * width + n * width if scan else 2 * n * max_tokens * width
    scratch = np.zeros(max(1, scratch_size), dtype=np.float32)
    logits = np.zeros(max(1, n * max_tokens * m["tags"] + n * m["pooled"]), dtype=np.float32)
    g = grid(n * max_tokens)
    if scan:
        passes: list[tuple[str, tuple[int, ...]]] = [("embed", g)]
        for l in range(layers):
            passes += [(f"gates{l}", g), (f"scan{l}", (n,)), (f"conv{l}", g)]
        passes += [("pool", (n,)), ("head", g), ("pooled", (n,))]
    else:
        passes = [("embed", g), *[(f"block{i}", g) for i in range(layers)], ("head", g), ("pool", (n,))]
    out = runner.run(
        {
            "params": Uniform(params),
            "table": table,
            "weights": weights.astype(np.float32),
            "rows": rows,
            "lengths": lengths,
            "scratch": scratch,
            "logits": logits,
        },
        passes,
        readback="logits",
    )
    tags = [out[s * max_tokens * m["tags"] : s * max_tokens * m["tags"] + int(lengths[s]) * m["tags"]].reshape(int(lengths[s]), m["tags"]) for s in range(n)]
    if m["pooled"] <= 0:
        return tags, None
    base = n * max_tokens * m["tags"]
    pooled = [out[base + s * m["pooled"] : base + (s + 1) * m["pooled"]].copy() for s in range(n)]
    return tags, pooled
