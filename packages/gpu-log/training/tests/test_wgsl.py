"""Runs src/shader.wgsl on a real WebGPU adapter (lavapipe on CI/dev boxes, any GPU otherwise)
and checks its logits against model/fixtures.json, i.e. against the CPU reference path, using
the same buffer layout, offset table and dispatch grid as src/gpu.ts. Skipped when wgpu or an
adapter is unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpu_log.features import featurize
from gpu_log.model import forward_numpy

PKG = Path(__file__).resolve().parents[2]
MANIFEST = PKG / "model" / "manifest.json"
WEIGHTS = PKG / "model" / "weights.txt"
FIXTURES = PKG / "model" / "fixtures.json"
SHADER = PKG / "src" / "shader.wgsl"

wgpu = pytest.importorskip("wgpu")

ENTRIES = ["embed", "block0", "block1", "block2", "block3", "block4", "heads"]
H, GRID_X = 64, 32768


def offset_table(manifest: dict) -> np.ndarray:
    names = ["embed", "proj_w", "proj_b"]
    for i in range(manifest["blocks"]):
        names += [f"block{i}_w1", f"block{i}_b1", f"block{i}_w2", f"block{i}_b2"]
    names += ["head_h_w", "head_h_b", "head_tag_w", "head_tag_b", "head_kind_w", "head_kind_b"]
    offsets = {t["name"]: t["offset"] for t in manifest["tensors"]}
    return np.asarray([offsets[n] for n in names], dtype=np.uint32)


def decode_weights(encoded: str, tensors: list[dict]) -> np.ndarray:
    alphabet = "".join(chr(c) for c in range(35, 127) if chr(c) not in "\\\"'`")[:64]
    lookup = np.zeros(128, dtype=np.float32)
    for i, ch in enumerate(alphabet):
        lookup[ord(ch)] = i - 32
    codes = np.frombuffer(encoded.encode("ascii"), dtype=np.uint8)
    out = lookup[codes]
    for t in tensors:
        out[t["offset"] : t["offset"] + t["length"]] *= np.float32(t["scale"])
    return out.astype(np.float32)


def dequantized_tensors(manifest: dict, w: np.ndarray) -> dict[str, np.ndarray]:
    return {e["name"]: w[e["offset"] : e["offset"] + e["length"]].reshape(e["shape"]) for e in manifest["tensors"]}


@pytest.fixture(scope="module")
def device():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no WebGPU adapter: {e}")
    return adapter.request_device_sync()


def run_shader(device, manifest: dict, weights: np.ndarray, lines: list[list[list[int]]]) -> list[np.ndarray]:
    """Packs several lines into one dispatch (like index.ts) and returns [L_i, T + K] logits per line."""
    rows = [r for line in lines for r in line]
    n = len(rows)
    W = len(manifest["labels"]) + len(manifest["kinds"])
    line_id = np.concatenate([np.full(len(line), i, dtype=np.uint32) for i, line in enumerate(lines)])
    module = device.create_shader_module(code=SHADER.read_text())
    pipelines = {e: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e}) for e in ENTRIES}
    U = wgpu.BufferUsage
    bufs = [
        device.create_buffer_with_data(data=np.asarray([n, 0, 0, 0], dtype=np.uint32), usage=U.UNIFORM | U.COPY_DST),
        device.create_buffer_with_data(data=offset_table(manifest), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.asarray(rows, dtype=np.uint32).ravel(), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=line_id, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=weights, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * H, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * H, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * W, dtype=np.float32), usage=U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
    # Auto layouts are exclusive to their pipeline: one bind group per pass (as in runtime/program.ts).
    bind_groups = {e: device.create_bind_group(layout=pipelines[e].get_bind_group_layout(0), entries=entries) for e in ENTRIES}
    grid = (min(n, GRID_X), -(-n // GRID_X), 1)
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    for entry in ENTRIES:
        cpass.set_pipeline(pipelines[entry])
        cpass.set_bind_group(0, bind_groups[entry])
        cpass.dispatch_workgroups(*grid)
    cpass.end()
    device.queue.submit([encoder.finish()])
    out = np.frombuffer(device.queue.read_buffer(bufs[7]), dtype=np.float32)[: n * W].reshape(n, W)
    result = []
    at = 0
    for line in lines:
        result.append(out[at : at + len(line)])
        at += len(line)
    return result


def test_wgsl_matches_fixtures(device) -> None:
    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    fixtures = [c for c in json.loads(FIXTURES.read_text()) if c["features"]]
    assert len(fixtures) >= 20
    got = run_shader(device, manifest, weights, [c["features"] for c in fixtures])
    worst = 0.0
    for case, logits in zip(fixtures, got, strict=True):
        expected = np.asarray(case["logits"], dtype=np.float32)
        worst = max(worst, float(np.abs(logits - expected).max()))
    assert worst < 1e-3, f"max |gpu - reference| = {worst}"


def test_wgsl_isolates_lines_in_a_packed_batch(device) -> None:
    """Packed lines must give the same logits as running each line alone (line_id masking)."""
    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    deq = dequantized_tensors(manifest, weights)
    texts = ["\tat com.example.Foo.bar(Foo.java:42)", "Jan 15 10:30:00 host sshd[1]: ok", "x"]
    lines = [featurize(t)[1] for t in texts]
    got = run_shader(device, manifest, weights, lines)
    for rows, logits in zip(lines, got, strict=True):
        ref = forward_numpy(deq, np.asarray(rows), blocks=manifest["blocks"])
        assert np.abs(logits - ref).max() < 1e-3
