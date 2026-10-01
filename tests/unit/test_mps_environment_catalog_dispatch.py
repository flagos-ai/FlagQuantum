"""Catalog dispatch contracts for fused MPS environment transfers."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.environment_dispatch as environment_dispatch
from flagquantum.simulation.mps.environment_dispatch import (
    _mps_environment_kernel_enabled,
    _mps_environment_kernel_match,
    _require_mps_environment_kernel,
)
from flagquantum.simulation.mps.site_kernels import (
    environment_transfer,
    reset_site_kernel_stats,
    site_kernel_stats,
)

pytestmark = pytest.mark.unit


def test_mps_environment_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_environment_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.environment.transfer_identity_z"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-004-A"
    assert implementation.symbol == "fused_mps_environment_transfer"
    assert implementation.directions == ("forward",)


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_environment_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_environment_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-004-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_environment_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_environment_kernel(
            device_type="cuda",
            dtype="complex128",
        )


def test_mps_environment_route_rejects_cpu_even_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "1")
    environment = torch.randn(2, 4, 4, dtype=torch.complex64)
    tensor = torch.randn(2, 4, 2, 4, dtype=torch.complex64)

    assert not _mps_environment_kernel_enabled(environment, tensor)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_route_enforces_evidenced_window(monkeypatch) -> None:
    environment = torch.randn(8, 16, 16, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(8, 16, 2, 16, device="cuda", dtype=torch.complex64)

    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    assert not _mps_environment_kernel_enabled(environment, tensor)

    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "1")
    assert _mps_environment_kernel_enabled(environment, tensor)
    noncontiguous = torch.randn(
        8, 16, 2, 16, device="cuda", dtype=torch.complex64
    ).transpose(1, 3)
    assert noncontiguous.shape == tensor.shape
    assert not noncontiguous.is_contiguous()
    assert not _mps_environment_kernel_enabled(environment, noncontiguous)

    large_environment = torch.randn(16, 32, 32, device="cuda", dtype=torch.complex64)
    large_tensor = torch.randn(16, 32, 2, 32, device="cuda", dtype=torch.complex64)
    assert not _mps_environment_kernel_enabled(large_environment, large_tensor)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_route_rejects_gradients(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "1")
    environment = torch.randn(
        2,
        4,
        4,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    tensor = torch.randn(2, 4, 2, 4, device="cuda", dtype=torch.complex64)

    assert not _mps_environment_kernel_enabled(environment, tensor)
    actual = environment_transfer(environment, tensor, z=True, compiled=False)
    actual.real.sum().backward()

    assert environment.grad is not None


@pytest.mark.parametrize("insert_z", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_product_path_uses_catalog(
    monkeypatch,
    insert_z: bool,
    compiled: bool,
) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "1")
    catalog_routes = []
    require_cataloged_kernel = environment_dispatch._require_mps_environment_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        environment_dispatch,
        "_require_mps_environment_kernel",
        capture_catalog_route,
    )
    environment = torch.randn(8, 16, 16, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(8, 16, 2, 16, device="cuda", dtype=torch.complex64)
    signs = tensor.real.new_tensor((1.0, -1.0) if insert_z else (1.0, 1.0))
    expected = torch.einsum(
        "bij,bipr,bjps->brs",
        environment,
        tensor.conj(),
        tensor * signs.reshape(1, 1, 2, 1),
    )
    reset_site_kernel_stats(clear_cache=True)

    actual = environment_transfer(
        environment,
        tensor,
        z=insert_z,
        compiled=compiled,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=6e-5)
    assert site_kernel_stats()["triton_environment_transfer_calls"] == 1
    assert catalog_routes == ["FQKI-TRITON-MPS-004-A"]
