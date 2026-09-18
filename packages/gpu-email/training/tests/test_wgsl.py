"""Runs src/shader.wgsl on a real WebGPU adapter (lavapipe on CI/dev boxes, any GPU
otherwise) and checks the logits against model/fixtures.json, i.e. against the PyTorch
reference that the TypeScript CPU path is also checked against. Skipped when wgpu or an
adapter is unavailable.

Mirrors src/gpu.ts: same buffers, same offsets table, same dispatch sizes, one bind
group per pipeline (auto layouts are exclusive to their pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[2]
MANIFEST = PKG / "model" / "manifest.json"
WEIGHTS = PKG / "model" / "weights.txt"
FIXTURES = PKG / "model" / "fixtures.json"
SHADER = PKG / "src" / "shader.wgsl"

wgpu = pytest.importorskip("wgpu")

DIM, HEAD, KINDS, BIO, LAYERS = 48, 64, 8, 15, 6
LOGITS = KINDS + BIO
ENTRIES = ["embed", *[f"conv{i}" for i in range(LAYERS)], "head"]


def decode_weights(encoded: str, tensors: list[dict]) -> np.ndarray:
    alphabet = "".join(chr(c) for c in range(35, 127) if chr(c) not in "\\\"'`")[:64]
    lookup = {ch: i - 32 for i, ch in enumerate(alphabet)}
    out = np.zeros(len(encoded), dtype=np.float32)
    for t in tensors:
        for i in range(t["length"]):
            out[t["offset"] + i] = np.float32(lookup[encoded[t["offset"] + i]] * t["scale"])
    return out


@pytest.fixture(scope="module")
def device():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no WebGPU adapter: {e}")
    return adapter.request_device_sync()


@pytest.fixture(scope="module")
def model():
    manifest = json.loads(MANIFEST.read_text())
    return manifest, decode_weights(WEIGHTS.read_text(), manifest["tensors"])


def run_shader(device, manifest: dict, weights: np.ndarray, rows: list[list[int]]) -> np.ndarray:
    n = len(rows)
    slots = manifest["slots"]
    assert manifest["hidden"] == DIM and manifest["head"] == HEAD and len(manifest["dilations"]) == LAYERS
    off = {t["name"]: t["offset"] for t in manifest["tensors"]}
    offsets = np.zeros(7 + 3 * LAYERS, dtype=np.uint32)
    offsets[:7] = [off["emb"], off["head.w"], off["head.b"], off["kind.w"], off["kind.b"], off["bio.w"], off["bio.b"]]
    for l in range(LAYERS):
        offsets[7 + 3 * l : 10 + 3 * l] = [off[f"conv{l}.w"], off[f"conv{l}.b"], manifest["dilations"][l]]
    params = np.asarray([n, slots, LAYERS, 0], dtype=np.uint32)
    module = device.create_shader_module(code=SHADER.read_text())
    pipelines = {e: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e}) for e in ENTRIES}
    U = wgpu.BufferUsage
    bufs = [
        device.create_buffer_with_data(data=params, usage=U.UNIFORM | U.COPY_DST),
        device.create_buffer_with_data(data=offsets, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.asarray(rows, dtype=np.uint32).ravel(), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=weights, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros((LAYERS + 1) * n * DIM, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * LOGITS, dtype=np.float32), usage=U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
    bind_groups = {e: device.create_bind_group(layout=pipelines[e].get_bind_group_layout(0), entries=entries) for e in ENTRIES}
    channel_groups = -(-(n * DIM) // 64)
    token_groups = -(-n // 64)
    dispatch = [("embed", channel_groups), *[(f"conv{l}", channel_groups) for l in range(LAYERS)], ("head", token_groups)]
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    for entry, groups in dispatch:
        cpass.set_pipeline(pipelines[entry])
        cpass.set_bind_group(0, bind_groups[entry])
        cpass.dispatch_workgroups(groups, 1, 1)
    cpass.end()
    device.queue.submit([encoder.finish()])
    out = np.frombuffer(device.queue.read_buffer(bufs[5]), dtype=np.float32)
    return out[: n * LOGITS].reshape(n, LOGITS)


def test_wgsl_matches_fixtures(device, model) -> None:
    manifest, weights = model
    fixtures = json.loads(FIXTURES.read_text())
    worst = 0.0
    checked = 0
    for case in fixtures:
        if not case["features"]:
            continue
        got = run_shader(device, manifest, weights, case["features"])
        expected = np.asarray(case["logits"], dtype=np.float32)
        worst = max(worst, float(np.abs(got - expected).max()))
        checked += 1
    assert checked >= 20
    assert worst < 1e-3, f"max |gpu - torch| = {worst}"


def test_wgsl_long_input(device, model) -> None:
    """A multi-thousand-token input exercises dilation 32 across many workgroups."""
    from gpu_email.features import featurize
    from gpu_email.export import dequantized_model, load_model, logits_for

    manifest, weights = model
    text = ("Thanks for the update, that works for me.\n" * 60) + "\nOn Mon, Jan 5, 2024 at 3:14 PM Bob <bob@example.com> wrote:\n" + ("> quoted line\n" * 40)
    rows = featurize(text)
    assert len(rows) > 512
    got = run_shader(device, manifest, weights, rows)
    try:
        q = dequantized_model(load_model()[0])
    except FileNotFoundError:
        pytest.skip("no checkpoint in runs/")
    _, expected = logits_for(q, text)
    assert np.abs(got - expected).max() < 1e-3
