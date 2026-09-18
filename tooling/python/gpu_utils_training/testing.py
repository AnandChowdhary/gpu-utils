"""pytest helpers. Add to a package's conftest.py::

    from gpu_utils_training.testing import wgpu_device  # noqa: F401

and take ``wgpu_device`` as a fixture in WGSL parity tests; they skip cleanly when no
adapter exists (CI installs Mesa's lavapipe so they run there).
"""

from __future__ import annotations

from typing import Any

import pytest

from .wgsl import NoAdapterError, get_device


@pytest.fixture(scope="session")
def wgpu_device() -> Any:
    try:
        return get_device()
    except NoAdapterError as exc:
        pytest.skip(str(exc))


def has_adapter() -> bool:
    try:
        get_device()
        return True
    except NoAdapterError:
        return False
