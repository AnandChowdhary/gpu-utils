"""Runs src/shader.wgsl on a real WebGPU adapter (lavapipe on CI/dev boxes, any GPU otherwise)
and checks the logits against model/fixtures.json, i.e. against the PyTorch/CPU reference.
Skipped when wgpu or an adapter is unavailable. Buffer layout, uniform block and dispatch
sizes mirror src/gpu.ts exactly."""

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

ENTRIES = ["embed", "gates", "scan_local", "scan_fixup", "pool", "head"]
TENSORS = ["emb", "wa_f", "ba_f", "wu_f", "bu_f", "wa_b", "ba_b", "wu_b", "bu_b", "conv", "bc", "w1", "wg", "b1", "w2", "b2"]
D, W, H, CHUNK = 24, 72, 32, 256


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


def run_shader(device, manifest: dict, weights: np.ndarray, rows: list[list[int]]) -> np.ndarray:
    n = len(rows)
    out_cols = manifest["out"]
    width = manifest["width"]
    offsets = {t["name"]: t["offset"] for t in manifest["tensors"]}
    params = np.zeros(24, dtype=np.uint32)
    params[:4] = [n, -(-n // CHUNK), out_cols, width]
    params[4 : 4 + len(TENSORS)] = [offsets[name] for name in TENSORS]
    module = device.create_shader_module(code=SHADER.read_text())
    pipelines = {e: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e}) for e in ENTRIES}
    U = wgpu.BufferUsage
    bufs = [
        device.create_buffer_with_data(data=params, usage=U.UNIFORM | U.COPY_DST),
        device.create_buffer_with_data(data=np.asarray(rows, dtype=np.uint32).ravel(), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=weights, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * D + n * W + H, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(6 * n * D, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * out_cols, dtype=np.float32), usage=U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
    # Auto layouts are exclusive to their pipeline: one bind group per pass (as in runtime/program.ts).
    bind_groups = {e: device.create_bind_group(layout=pipelines[e].get_bind_group_layout(0), entries=entries) for e in ENTRIES}
    dispatch = [
        ("embed", (-(-(n * D) // 64), 1, 1)),
        ("gates", (-(-(2 * n * D) // 64), 1, 1)),
        ("scan_local", (-(-n // CHUNK), 2 * D, 1)),
        ("scan_fixup", (-(-(2 * n * D) // 64), 1, 1)),
        ("pool", (1, 1, 1)),
        ("head", (-(-n // 64), 1, 1)),
    ]
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    for entry, groups in dispatch:
        cpass.set_pipeline(pipelines[entry])
        cpass.set_bind_group(0, bind_groups[entry])
        cpass.dispatch_workgroups(*groups)
    cpass.end()
    device.queue.submit([encoder.finish()])
    out = np.frombuffer(device.queue.read_buffer(bufs[5]), dtype=np.float32)
    return out[: n * out_cols].reshape(n, out_cols)


def test_wgsl_matches_fixtures(device) -> None:
    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    fixtures = json.loads(FIXTURES.read_text())
    worst = 0.0
    checked = 0
    for case in fixtures:
        if not case["features"]:
            continue
        got = run_shader(device, manifest, weights, case["features"])
        expected = np.asarray(case["logits"], dtype=np.float32).reshape(got.shape)
        worst = max(worst, float(np.abs(got - expected).max()))
        checked += 1
    assert checked >= 20
    assert worst < 1e-3, f"max |gpu - torch| = {worst}"


def test_wgsl_long_input_crosses_chunks(device) -> None:
    """Inputs longer than one scan chunk exercise scan_fixup; compare against the PyTorch model."""
    import torch

    from gpu_tailwind.export import dequantized
    from gpu_tailwind.features import featurize
    from gpu_tailwind.model import Tagger

    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    text = ("card with rounded corners, subtle shadow, blue on hover, hidden on mobile, " * 12).strip()
    rows = featurize(text)
    assert len(rows) > CHUNK
    got = run_shader(device, manifest, weights, rows)
    # reference: torch model with the same dequantized weights
    model = Tagger()
    t = {e["name"]: weights[e["offset"] : e["offset"] + e["length"]].reshape(e["shape"]) for e in manifest["tensors"]}
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.from_numpy(np.ascontiguousarray(t[name])))
    model.quant = False
    model.eval()
    with torch.no_grad():
        ref = model(torch.tensor([rows]), torch.ones(1, len(rows), dtype=torch.bool))[0].numpy()
    assert dequantized is not None
    assert np.abs(got - ref).max() < 1e-3
