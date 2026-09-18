import numpy as np
import torch

from gpu_utils_training.batch import collate, pack_rows, pad_labels
from gpu_utils_training.kernels import grid, tagger_params
from gpu_utils_training.wgsl import Uniform, WgslRunner

SHADER = """
struct P { n: u32, a: u32, b: u32, c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn double(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < p.n) { y[id.x] = x[id.x] * 2.0; }
}
// Every entry point must reference every binding (one bind group per pipeline).
@compute @workgroup_size(64) fn add_one(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = x[0];
  if (id.x < p.n) { y[id.x] = y[id.x] + 1.0; }
}
"""


def test_pack_rows_and_collate() -> None:
    rows, lengths = pack_rows([[[1, 2, 3], [4]], [[5]]], slots=2, padding_id=99)
    assert rows.shape == (2, 2, 2) and rows.dtype == np.uint32
    assert rows[0].tolist() == [[1, 2], [4, 99]] and rows[1].tolist() == [[5, 99], [99, 99]]
    assert lengths.tolist() == [2, 1]
    r, mask = collate([[[1]], []], 1, 0)
    assert r.dtype == torch.int64 and mask.tolist() == [[True], [False]]
    assert pad_labels([[1, 2], [3]], 3).tolist() == [[1, 2, -100], [3, -100, -100]]


def test_params_and_grid() -> None:
    p = tagger_params(batch=2, max_tokens=5, slots=3, padding=9, embed=8, hidden=8, head=16, tags=4, pooled=0, layers=1, taps=5)
    assert p.tolist()[:11] == [2, 5, 3, 9, 8, 8, 16, 4, 0, 1, 5] and len(p) == 16
    assert grid(10) == (10, 1) and grid(70000) == (32768, 3) and grid(0) == (1, 1)


def test_wgsl_runner_multi_pass(wgpu_device) -> None:
    runner = WgslRunner(SHADER, ["double", "add_one"], wgpu_device)
    x = np.arange(5, dtype=np.float32)
    out = runner.run(
        {"p": Uniform(np.array([5, 0, 0, 0], dtype=np.uint32)), "x": x, "y": np.zeros(5, dtype=np.float32)},
        [("double", (1,)), {"entry": "add_one", "workgroups": [1]}],
        readback="y",
    )
    assert out.tolist() == [1, 3, 5, 7, 9]
