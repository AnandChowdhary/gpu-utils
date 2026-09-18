"""Runs src/shader.wgsl on a real WebGPU adapter (lavapipe on CI/dev boxes, any GPU otherwise)
and checks the logits against model/fixtures.json (= the PyTorch / CPU reference numbers).

Mirrors src/gpu.ts exactly: one flat token stream per batch, an `info` buffer with tensor
offsets, state offsets, the sequence table and the token→sequence map, and one bind group
per pipeline (auto layouts are exclusive to their pipeline, as in runtime/program.ts).
Skipped when wgpu or an adapter is unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gpu_utils_training.features import tokenize

from gpu_cite.features import WIDTH, featurize_tokens
from gpu_cite.model import CiteTagger

PKG = Path(__file__).resolve().parents[2]
MANIFEST = PKG / "model" / "manifest.json"
WEIGHTS = PKG / "model" / "weights.txt"
FIXTURES = PKG / "model" / "fixtures.json"
SHADER = PKG / "src" / "shader.wgsl"

wgpu = pytest.importorskip("wgpu")

ENTRIES = ["embed", "gates1", "scan1", "conv", "gates2", "scan2", "pool", "head_hidden", "head_out", "type_hidden", "type_out"]
# Must match TENSOR_ORDER in src/gpu.ts and the M_* constants in shader.wgsl.
TENSOR_ORDER = [
    "emb", "s1f_wa", "s1f_ba", "s1f_wb", "s1f_bb", "s1b_wa", "s1b_ba", "s1b_wb", "s1b_bb", "conv_w", "conv_b",
    "s2f_wa", "s2f_ba", "s2f_wb", "s2f_bb", "s2b_wa", "s2b_ba", "s2b_wb", "s2b_bb",
    "w1", "b1", "wt", "bt", "wp", "bp", "wc", "bc", "wd", "bd",
]
STATE_SLOTS = 9
CHUNK = 256


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
def model_data() -> tuple[dict, np.ndarray]:
    manifest = json.loads(MANIFEST.read_text())
    return manifest, decode_weights(WEIGHTS.read_text(), manifest["tensors"])


def run_shader(device, manifest: dict, weights: np.ndarray, batch: list[list[list[int]]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run one batch of sequences (each a list of feature rows). Returns (tags, parts, type) per sequence."""
    E, H, HEAD = manifest["embed"], manifest["hidden"], manifest["head"]
    C = 2 * H
    K, P, T = len(manifest["labels"]), len(manifest["nameparts"]), len(manifest["types"])
    W = manifest["featureWidth"]
    S = len(batch)
    lengths = [len(rows) for rows in batch]
    N = sum(lengths)
    assert N > 0
    features = np.zeros(N * W, dtype=np.uint32)
    info = np.zeros(len(TENSOR_ORDER) + STATE_SLOTS + 2 * S + N, dtype=np.uint32)
    seq_base = len(TENSOR_ORDER) + STATE_SLOTS
    cursor = 0
    for s, rows in enumerate(batch):
        info[seq_base + 2 * s] = cursor
        info[seq_base + 2 * s + 1] = lengths[s]
        for row in rows:
            features[cursor * W : (cursor + 1) * W] = row
            info[seq_base + 2 * S + cursor] = s
            cursor += 1
    offsets = {t["name"]: t["offset"] for t in manifest["tensors"]}
    for i, name in enumerate(TENSOR_ORDER):
        info[i] = offsets[name]
    state_sizes = [N * E, N * C * 2, N * C, N * C, N * C * 2, N * C, S * 2 * C, N * HEAD, S * H]
    total = 0
    for i, size in enumerate(state_sizes):
        info[len(TENSOR_ORDER) + i] = total
        total += size
    logit_total = N * K + N * P + S * T
    params = np.array([N, S, E, H, HEAD, K, P, T, W, 0, 0, 0], dtype=np.uint32)

    module = device.create_shader_module(code=SHADER.read_text())
    pipelines = {e: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e}) for e in ENTRIES}
    U = wgpu.BufferUsage
    bufs = [
        device.create_buffer_with_data(data=params, usage=U.UNIFORM | U.COPY_DST),
        device.create_buffer_with_data(data=info, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=features, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=weights, usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(total, dtype=np.float32), usage=U.STORAGE | U.COPY_DST),
        device.create_buffer_with_data(data=np.zeros(logit_total, dtype=np.float32), usage=U.STORAGE | U.COPY_DST | U.COPY_SRC),
    ]
    entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
    bind_groups = {e: device.create_bind_group(layout=pipelines[e].get_bind_group_layout(0), entries=entries) for e in ENTRIES}

    def groups(count: int) -> tuple[int, int, int]:
        return (max(1, -(-count // 64)), 1, 1)

    dispatch = [
        ("embed", groups(N * E)),
        ("gates1", groups(N * C)),
        ("scan1", (S * C, 1, 1)),
        ("conv", groups(N * C)),
        ("gates2", groups(N * C)),
        ("scan2", (S * C, 1, 1)),
        ("pool", groups(S * C)),
        ("head_hidden", groups(N * HEAD)),
        ("head_out", groups(N * (K + P))),
        ("type_hidden", groups(S * H)),
        ("type_out", groups(S * T)),
    ]
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    for entry, wg in dispatch:
        cpass.set_pipeline(pipelines[entry])
        cpass.set_bind_group(0, bind_groups[entry])
        cpass.dispatch_workgroups(*wg)
    cpass.end()
    device.queue.submit([encoder.finish()])
    out = np.frombuffer(device.queue.read_buffer(bufs[5]), dtype=np.float32)
    results = []
    cursor = 0
    for s in range(S):
        n = lengths[s]
        tags = out[cursor * K : (cursor + n) * K].reshape(n, K)
        parts = out[N * K + cursor * P : N * K + (cursor + n) * P].reshape(n, P)
        ty = out[N * K + N * P + s * T : N * K + N * P + (s + 1) * T]
        results.append((tags, parts, ty))
        cursor += n
    return results


def test_wgsl_matches_fixtures_one_sequence_per_dispatch(device, model_data) -> None:
    manifest, weights = model_data
    fixtures = json.loads(FIXTURES.read_text())
    worst = 0.0
    checked = 0
    for case in fixtures:
        if not case["rows"]:
            continue
        [(tags, parts, ty)] = run_shader(device, manifest, weights, [case["rows"]])
        worst = max(
            worst,
            float(np.abs(tags - np.asarray(case["tags"], dtype=np.float32).reshape(tags.shape)).max()),
            float(np.abs(parts - np.asarray(case["parts"], dtype=np.float32).reshape(parts.shape)).max()),
            float(np.abs(ty - np.asarray(case["type"], dtype=np.float32)).max()),
        )
        checked += 1
    assert checked >= 20
    assert worst < 1e-3, f"max |gpu - torch| = {worst}"


def test_wgsl_batched_dispatch_matches_fixtures(device, model_data) -> None:
    """All fixtures in one command buffer: exercises the sequence table and token→sequence map."""
    manifest, weights = model_data
    fixtures = [c for c in json.loads(FIXTURES.read_text()) if c["rows"]]
    results = run_shader(device, manifest, weights, [c["rows"] for c in fixtures])
    worst = 0.0
    for case, (tags, parts, ty) in zip(fixtures, results, strict=True):
        worst = max(
            worst,
            float(np.abs(tags - np.asarray(case["tags"], dtype=np.float32).reshape(tags.shape)).max()),
            float(np.abs(parts - np.asarray(case["parts"], dtype=np.float32).reshape(parts.shape)).max()),
            float(np.abs(ty - np.asarray(case["type"], dtype=np.float32)).max()),
        )
    assert worst < 1e-3, f"max |gpu - torch| = {worst}"


def torch_reference(manifest: dict, weights: np.ndarray, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PyTorch forward with the exported (int6-decoded) weights, no fake-quant."""
    model = CiteTagger(embed=manifest["embed"], hidden=manifest["hidden"], head=manifest["head"])
    entries = {t["name"]: t for t in manifest["tensors"]}
    with torch.no_grad():
        for name, param in model.export_tensors().items():
            e = entries[name]
            param.copy_(torch.from_numpy(weights[e["offset"] : e["offset"] + e["length"]].reshape(e["shape"]).copy()))
    model.quant = False
    model.eval()
    with torch.no_grad():
        r = torch.as_tensor(rows, dtype=torch.long).unsqueeze(0)
        tags, parts, ty = model(r, torch.ones(1, len(rows)))
    return tags[0].numpy(), parts[0].numpy(), ty[0].numpy()


def test_wgsl_long_input_crosses_scan_chunks(device, model_data) -> None:
    """A reference longer than one 256-token scan chunk exercises the carry between chunks."""
    manifest, weights = model_data
    text = ("Smith, J., Doe, A. B., & Roe, C. (2019). A study of things. Journal of Stuff, 12(3), 45–67. " * 6).strip()
    toks = tokenize(text)
    rows = featurize_tokens(text, toks)
    assert len(rows) > CHUNK
    assert all(len(r) == WIDTH for r in rows)
    [(tags, parts, ty)] = run_shader(device, manifest, weights, [rows])
    ref_tags, ref_parts, ref_ty = torch_reference(manifest, weights, rows)
    assert np.abs(tags - ref_tags).max() < 1e-3
    assert np.abs(parts - ref_parts).max() < 1e-3
    assert np.abs(ty - ref_ty).max() < 1e-3
