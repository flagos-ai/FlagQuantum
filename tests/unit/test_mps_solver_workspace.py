from __future__ import annotations

import ctypes

import pytest

from flagquantum.runtime.executors.mps import solver_workspace

pytestmark = pytest.mark.unit


class _Library:
    pass


def test_probe_prefers_xgesvd_native_extension(monkeypatch) -> None:
    library = _Library()
    for symbol in (
        *solver_workspace._REQUIRED_XGESVD_SYMBOLS,
        *solver_workspace._REQUIRED_CLASSIC_GESVD_SYMBOLS,
    ):
        setattr(library, symbol, object())
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libcusolver.so")
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: library)

    result = solver_workspace.probe_solver_workspace_capabilities()

    assert result.cusolver_loadable
    assert result.classic_complex_gesvd_workspace_api
    assert result.xgesvd_device_host_workspace_api
    assert not result.torch_public_external_workspace
    assert result.native_extension_required
    assert result.recommended_path == "cpp_cuda_extension_cusolver_xgesvd"


def test_probe_fails_closed_when_cusolver_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: None)

    result = solver_workspace.probe_solver_workspace_capabilities()

    assert not result.cusolver_loadable
    assert not result.native_extension_required
    assert result.recommended_path == "torch_linalg_internal_workspace"
