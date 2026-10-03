"""Dependency-light contracts for the FlagTree TLE provider boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import flagquantum.kernels.flagtree as provider

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    provider.require_flagtree_tle_primitive.cache_clear()
    yield
    provider.require_flagtree_tle_primitive.cache_clear()


def test_tle_capability_maps_cuda_target_to_nvidia_registry(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    tle = SimpleNamespace(
        require_tle=lambda backend, primitive: calls.append((backend, primitive))
    )
    monkeypatch.setattr(
        provider,
        "triton_compiler_provenance",
        lambda: ("flagtree", "0.7.0", "flagtree", "resolved"),
    )
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda module, package=None: tle,
    )

    registry_backend = provider.require_flagtree_tle_primitive(
        "load",
        backend="cuda",
    )

    assert registry_backend == "nvidia"
    assert calls == [("nvidia", "load")]


def test_tle_capability_rejects_stock_triton_before_import(monkeypatch) -> None:
    monkeypatch.setattr(
        provider,
        "triton_compiler_provenance",
        lambda: ("triton", "3.7.1", "direct", "resolved"),
    )
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda *args, **kwargs: pytest.fail("TLE must not be imported"),
    )

    with pytest.raises(RuntimeError, match="flagtree distribution"):
        provider.require_flagtree_tle_primitive("load", backend="cuda")


def test_tle_capability_reports_missing_backend_primitive(monkeypatch) -> None:
    def reject(_backend: str, _primitive: str) -> None:
        raise RuntimeError("unsupported")

    monkeypatch.setattr(
        provider,
        "triton_compiler_provenance",
        lambda: ("flagtree", "0.7.0", "flagtree", "resolved"),
    )
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda module, package=None: SimpleNamespace(require_tle=reject),
    )

    with pytest.raises(RuntimeError, match="does not provide.*'load'.*'nvidia'"):
        provider.require_flagtree_tle_primitive("load", backend="cuda")


def test_tle_local_1q_rejects_cpu_before_import(monkeypatch) -> None:
    monkeypatch.setattr(
        provider,
        "require_flagtree_tle_primitive",
        lambda *args, **kwargs: pytest.fail("capability probe must not run"),
    )
    state = torch.zeros(1, 8, dtype=torch.complex64)
    matrix = torch.eye(2, dtype=torch.complex64)

    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        provider.apply_complex64_local_1q_tle(
            state,
            matrix,
            bit_position=0,
        )


def test_tle_control_pack_rejects_cpu_before_capability_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        provider,
        "require_flagtree_tle_primitive",
        lambda *args, **kwargs: pytest.fail("capability probe must not run"),
    )
    state = torch.zeros(1, 8, dtype=torch.complex64)

    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        provider.pack_complex64_control_one_tle(
            state,
            bit_position=0,
            compressed_start=0,
            compressed_end=4,
        )


def test_tle_control_unpack_rejects_cpu_before_capability_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        provider,
        "require_flagtree_tle_primitive",
        lambda *args, **kwargs: pytest.fail("capability probe must not run"),
    )
    packed = torch.zeros(1, 4, dtype=torch.complex64)
    output = torch.zeros(1, 8, dtype=torch.complex64)

    with pytest.raises(ValueError, match="matching contiguous CUDA complex64"):
        provider.unpack_complex64_control_one_tle(
            packed,
            output,
            bit_position=0,
            compressed_start=0,
        )
