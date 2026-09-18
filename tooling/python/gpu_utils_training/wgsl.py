"""Run WGSL compute shaders from Python through wgpu-py (any adapter, including Mesa's
llvmpipe software Vulkan) with the same binding conventions as
packages/runtime/src/program.ts: binding index = position in the buffer list, one bind
group per pipeline (so every entry point must statically reference every binding), all
passes in one command buffer, one readback.

    runner = WgslRunner(shader_source, ["embed", "head"])
    out = runner.run(
        {"params": Uniform(params_u32), "weights": w_f32, "rows": rows_u32, "logits": np.zeros(n, np.float32)},
        [("embed", (groups,)), ("head", (groups,))],
        readback="logits",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class NoAdapterError(RuntimeError):
    """Raised when no WebGPU adapter is available (wgpu missing, no Vulkan/Metal driver)."""


@dataclass(frozen=True)
class Uniform:
    """Marks a buffer as a uniform (``var<uniform>``) rather than storage."""

    data: np.ndarray


_device: Any = None


def get_device() -> Any:
    """Cached wgpu device; raises NoAdapterError when nothing can run compute shaders."""
    global _device
    if _device is not None:
        return _device
    try:
        import wgpu
    except ImportError as exc:  # pragma: no cover
        raise NoAdapterError(f"wgpu not installed: {exc}") from exc
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _device = adapter.request_device_sync()
    except Exception as exc:  # noqa: BLE001 - wgpu raises plain RuntimeErrors
        raise NoAdapterError(f"no WebGPU adapter: {exc}") from exc
    return _device


def adapter_info() -> dict[str, Any]:
    dev = get_device()
    try:
        return dict(dev.adapter.info)
    except Exception:  # noqa: BLE001
        return {}


def _pad16(b: bytes) -> bytes:
    n = max(16, -(-len(b) // 16) * 16)
    return b + bytes(n - len(b))


class WgslRunner:
    def __init__(self, shader_source: str, entry_points: list[str], device: Any = None) -> None:
        import wgpu

        self.wgpu = wgpu
        self.device = device or get_device()
        self.entries = list(entry_points)
        module = self.device.create_shader_module(code=shader_source)
        self.pipelines = {
            e: self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": e})
            for e in self.entries
        }

    def run(
        self,
        buffers: dict[str, np.ndarray | Uniform],
        passes: list[tuple[str, tuple[int, ...]] | dict[str, Any]],
        readback: str,
    ) -> np.ndarray:
        """Upload ``buffers`` in order, dispatch ``passes``, return ``readback`` as float32."""
        U = self.wgpu.BufferUsage
        names = list(buffers)
        gpu_buffers = []
        for name in names:
            spec = buffers[name]
            if isinstance(spec, Uniform):
                usage = U.UNIFORM | U.COPY_DST
                data = spec.data
            else:
                usage = U.STORAGE | U.COPY_DST | U.COPY_SRC
                data = spec
            gpu_buffers.append(self.device.create_buffer_with_data(label=name, data=_pad16(np.ascontiguousarray(data).tobytes()), usage=usage))
        entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(gpu_buffers)]
        bind_groups: dict[str, Any] = {}
        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        for p in passes:
            entry, groups = (p["entry"], tuple(p["workgroups"])) if isinstance(p, dict) else p
            if entry not in bind_groups:
                bind_groups[entry] = self.device.create_bind_group(layout=self.pipelines[entry].get_bind_group_layout(0), entries=entries)
            cpass.set_pipeline(self.pipelines[entry])
            cpass.set_bind_group(0, bind_groups[entry])
            g = tuple(groups) + (1,) * (3 - len(groups))
            cpass.dispatch_workgroups(*g)
        cpass.end()
        self.device.queue.submit([encoder.finish()])
        target = gpu_buffers[names.index(readback)]
        spec = buffers[readback]
        count = (spec.data if isinstance(spec, Uniform) else spec).size
        out = np.frombuffer(self.device.queue.read_buffer(target), dtype=np.float32)[:count].copy()
        for b in gpu_buffers:
            b.destroy()
        return out
