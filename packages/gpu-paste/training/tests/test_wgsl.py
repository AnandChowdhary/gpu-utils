"""Runs src/shader.wgsl on a real WebGPU adapter (lavapipe on CI/dev boxes, any GPU otherwise)
and checks the logits against model/fixtures.json, i.e. against the CPU reference path.
Skipped when wgpu or an adapter is unavailable."""

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

ENTRIES = ["embed", "gates", "scan_local", "scan_fixup", "mix", "pool", "head"]
TENSORS = ["embed", "gate_f.w", "gate_f.b", "gate_b.w", "gate_b.b", "mix.w", "mix.b", "head.w", "head.b", "out.w", "out.b", "kind1.w", "kind1.b", "kind2.w", "kind2.b"]
DIM, MIX, KH, CHUNK = 32, 48, 32, 256


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


def run_shader(device, manifest: dict, weights: np.ndarray, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    F = manifest["featureCount"]
    L = len(manifest["labels"])
    K = len(manifest["kinds"])
    offsets = {t["name"]: t["offset"] for t in manifest["tensors"]}
    params = np.zeros(24, dtype=np.uint32)
    params[:5] = [n, -(-n // CHUNK), L, K, F]
    params[5 : 5 + len(TENSORS)] = [offsets[name] for name in TENSORS]
    module = device.create_shader_module(code=SHADER.read_text())
    pipelines = {e: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e}) for e in ENTRIES}
    U = wgpu.BufferUsage
    bufs = [
        device.create_buffer_with_data(data=params, usage=U.UNIFORM | U.COPY_DST),
        device.create_buffer_with_data(data=np.asarray(rows, dtype=np.uint32).ravel(), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=weights, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * (DIM + MIX) + MIX + KH, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(6 * n * DIM, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(n * L + K, dtype=np.float32), usage=U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
    # Auto layouts are exclusive to their pipeline: one bind group per pass (as in runtime/program.ts).
    bind_groups = {e: device.create_bind_group(layout=pipelines[e].get_bind_group_layout(0), entries=entries) for e in ENTRIES}
    dispatch = [
        ("embed", (-(-(n * DIM) // 64), 1, 1)),
        ("gates", (-(-(2 * n * DIM) // 64), 1, 1)),
        ("scan_local", (-(-n // CHUNK), 2 * DIM, 1)),
        ("scan_fixup", (-(-(2 * n * DIM) // 64), 1, 1)),
        ("mix", (n, 1, 1)),
        ("pool", (1, 1, 1)),
        ("head", (n, 1, 1)),
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
    return out[: n * L].reshape(n, L), out[n * L : n * L + K]


def test_wgsl_matches_fixtures(device) -> None:
    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    fixtures = json.loads(FIXTURES.read_text())
    worst = 0.0
    checked = 0
    for case in fixtures:
        if not case["features"]:
            continue
        span, kind = run_shader(device, manifest, weights, case["features"])
        expected_span = np.asarray(case["span"], dtype=np.float32).reshape(span.shape)
        worst = max(worst, float(np.abs(span - expected_span).max()), float(np.abs(kind - np.asarray(case["kind"], dtype=np.float32)).max()))
        checked += 1
    assert checked >= 20
    assert worst < 1e-3, f"max |gpu - torch| = {worst}"


def test_wgsl_long_input_crosses_chunks(device) -> None:
    """Inputs longer than one scan chunk exercise scan_fixup; compare against a NumPy CPU reference."""
    from gpu_paste.features import featurize

    manifest = json.loads(MANIFEST.read_text())
    weights = decode_weights(WEIGHTS.read_text(), manifest["tensors"])
    text = ("Invoice from Acme Corp for $12.50, call +1 555 0100 by Friday. " * 30).strip()
    rows = featurize(text)
    assert len(rows) > CHUNK
    span, kind = run_shader(device, manifest, weights, rows)
    ref_span, ref_kind = cpu_reference(manifest, weights, rows)
    assert np.abs(span - ref_span).max() < 1e-3
    assert np.abs(kind - ref_kind).max() < 1e-3


def cpu_reference(manifest: dict, w: np.ndarray, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    t = {e["name"]: w[e["offset"] : e["offset"] + e["length"]].reshape(e["shape"]) for e in manifest["tensors"]}
    x = t["embed"][np.asarray(rows)].sum(axis=1)  # [n, D]
    n = len(rows)

    def scan(wname: str, bname: str, backward: bool) -> np.ndarray:
        pre = x @ t[wname].T + t[bname]
        a = 1 / (1 + np.exp(-pre[:, :DIM]))
        b = (1 - a) * np.tanh(pre[:, DIM:])
        h = np.zeros_like(a)
        prev = np.zeros(DIM, dtype=np.float32)
        order = range(n - 1, -1, -1) if backward else range(n)
        for i in order:
            prev = a[i] * prev + b[i]
            h[i] = prev
        return h

    hf = scan("gate_f.w", "gate_f.b", False)
    hb = scan("gate_b.w", "gate_b.b", True)
    m = np.maximum(np.concatenate([x, hf, hb], axis=1) @ t["mix.w"].T + t["mix.b"], 0)
    g = m.mean(axis=0)
    s = np.maximum(np.concatenate([m, np.tile(g, (n, 1))], axis=1) @ t["head.w"].T + t["head.b"], 0)
    span = s @ t["out.w"].T + t["out.b"]
    kh = np.maximum(g @ t["kind1.w"].T + t["kind1.b"], 0)
    kind = kh @ t["kind2.w"].T + t["kind2.b"]
    return span.astype(np.float32), kind.astype(np.float32)
