"""Executes src/shader.wgsl through wgpu-py (any adapter, including Mesa's llvmpipe
software Vulkan) and checks it against the exported fixtures. Skips without an adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[2]
SHADER = PACKAGE / "src" / "shader.wgsl"
MODEL = PACKAGE / "model"
NAMES = ["embedding", "encoder_bias", "convolution", "gate_weight", "gate_bias", "candidate_weight",
         "candidate_bias", "combine_weight", "combine_bias", "global_weight", "global_bias",
         "head_gate_weight", "head_gate_bias", "head_hidden_weight", "head_hidden_bias",
         "output_weight", "output_bias"]


def _device():
    wgpu = pytest.importorskip("wgpu")
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        return wgpu, adapter.request_device_sync()
    except Exception as exc:  # pragma: no cover - no adapter on this machine
        pytest.skip(f"no WebGPU adapter: {exc}")


def run_kernel(wgpu, device, weights: np.ndarray, manifest: dict, batch: list[list[list[int]]]) -> list[np.ndarray]:
    slots, padding = manifest["slots"], manifest["featureRows"]
    outs = len(manifest["labels"]) + 1
    hidden = manifest["hidden"]
    n = len(batch)
    max_tokens = max(1, max(len(rows) for rows in batch))
    rows = np.full((n, max_tokens, slots), padding, dtype=np.uint32)
    lengths = np.zeros(n, dtype=np.uint32)
    for s, r in enumerate(batch):
        lengths[s] = len(r)
        for t, row in enumerate(r):
            rows[s, t, : len(row)] = row
    offsets = np.zeros(20, dtype=np.uint32)
    by_name = {t["name"]: t for t in manifest["tensors"]}
    for i, name in enumerate(NAMES):
        offsets[i] = by_name[name]["offset"]
    params = np.array([n, max_tokens, slots, outs, hidden, manifest["headGate"], padding, 0], dtype=np.uint32)
    U = wgpu.BufferUsage
    mk = lambda arr, usage: device.create_buffer_with_data(data=np.ascontiguousarray(arr).tobytes(), usage=usage)
    scratch = np.zeros(max(1, 4 * n * max_tokens * hidden), dtype=np.float32)
    logits = np.zeros(max(1, n * max_tokens * outs), dtype=np.float32)
    bufs = [
        mk(params, U.UNIFORM | U.COPY_DST),
        mk(offsets, U.UNIFORM | U.COPY_DST),
        mk(weights.astype(np.float32), U.STORAGE | U.COPY_DST),
        mk(rows, U.STORAGE | U.COPY_DST),
        mk(lengths, U.STORAGE | U.COPY_DST),
        mk(scratch, U.STORAGE | U.COPY_DST),
        mk(logits, U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    module = device.create_shader_module(code=SHADER.read_text())
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "tag"})
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)],
    )
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipeline)
    p.set_bind_group(0, bind)
    p.dispatch_workgroups(n, 1, 1)
    p.end()
    device.queue.submit([enc.finish()])
    out = np.frombuffer(device.queue.read_buffer(bufs[-1]), dtype=np.float32).reshape(n, max_tokens, outs)
    return [out[s, : len(r)].copy() for s, r in enumerate(batch)]


def decode_weights(manifest: dict) -> np.ndarray:
    from gpu_utils_training.quant import ALPHABET

    lookup = {ch: i - 32 for i, ch in enumerate(ALPHABET)}
    encoded = (MODEL / "weights.txt").read_text()
    out = np.zeros(len(encoded), dtype=np.float32)
    for t in manifest["tensors"]:
        seg = encoded[t["offset"] : t["offset"] + t["length"]]
        out[t["offset"] : t["offset"] + t["length"]] = (np.array([lookup[c] for c in seg], dtype=np.float64) * t["scale"]).astype(np.float32)
    return out


def test_wgsl_matches_fixtures_batched() -> None:
    wgpu, device = _device()
    manifest = json.loads((MODEL / "manifest.json").read_text())
    fixtures = json.loads((MODEL / "fixtures.json").read_text())
    weights = decode_weights(manifest)
    batch = [f["rows"] for f in fixtures if f["rows"]]
    got = run_kernel(wgpu, device, weights, manifest, batch)
    worst = 0.0
    for f, g in zip([f for f in fixtures if f["rows"]], got):
        worst = max(worst, float(np.abs(g - np.array(f["logits"], dtype=np.float32)).max()))
    print(f"WGSL vs fixtures over {len(batch)} phrases in one dispatch: max |Δ| = {worst:.2e}")
    assert worst < 1e-4
