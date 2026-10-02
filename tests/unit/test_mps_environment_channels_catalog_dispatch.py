"""Catalog dispatch contracts for fused MPS environment channels."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.environment_dispatch as environment_dispatch
from flagquantum.simulation.mps.environment_dispatch import (
    _mps_environment_channels_kernel_enabled,
    _mps_environment_channels_kernel_match,
    _mps_environment_channels_rollout_enabled,
    _require_mps_environment_channels_kernel,
)
from flagquantum.simulation.mps.site_kernels import (
    environment_transfer_channels,
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)

pytestmark = pytest.mark.unit


def test_mps_environment_channels_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_environment_channels_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.environment.transfer_channels"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-005-A"
    assert implementation.symbol == "fused_mps_environment_channels"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_environment_channels_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_environment_channels_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-005-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_environment_channels_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_environment_channels_kernel(
            device_type="cuda",
            dtype="complex128",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_mps_environment_channels_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    assert _mps_environment_channels_rollout_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", disabled)
    assert not _mps_environment_channels_rollout_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "1")
    assert _mps_environment_channels_rollout_enabled()


def test_mps_environment_channels_route_rejects_cpu_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    channels = torch.randn(5, 2, 4, 4, dtype=torch.complex64)
    tensor = torch.randn(2, 4, 2, 4, dtype=torch.complex64)

    assert not _mps_environment_channels_kernel_enabled(channels, tensor)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_channels_route_enforces_evidenced_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = torch.randn(16, 8, 16, 16, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(8, 16, 2, 16, device="cuda", dtype=torch.complex64)

    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    assert _mps_environment_channels_kernel_enabled(channels, tensor)

    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "0")
    assert not _mps_environment_channels_kernel_enabled(channels, tensor)
    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    assert not _mps_environment_channels_kernel_enabled(
        channels.transpose(-2, -1),
        tensor,
    )

    too_many_channels = torch.randn(33, 1, 4, 4, device="cuda", dtype=torch.complex64)
    small_tensor = torch.randn(1, 4, 2, 4, device="cuda", dtype=torch.complex64)
    assert not _mps_environment_channels_kernel_enabled(
        too_many_channels,
        small_tensor,
    )

    too_much_work = torch.randn(32, 8, 16, 16, device="cuda", dtype=torch.complex64)
    assert not _mps_environment_channels_kernel_enabled(too_much_work, tensor)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_channels_route_rejects_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    channels = torch.randn(
        5,
        2,
        4,
        4,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    tensor = torch.randn(2, 4, 2, 4, device="cuda", dtype=torch.complex64)

    assert not _mps_environment_channels_kernel_enabled(channels, tensor)
    actual = environment_transfer_channels(channels, tensor, compiled=False)
    actual.real.sum().backward()

    assert channels.grad is not None


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_channels_product_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
    compiled: bool,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ENVIRONMENT", raising=False)
    catalog_routes: list[str] = []
    require_cataloged_kernel = (
        environment_dispatch._require_mps_environment_channels_kernel
    )

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        environment_dispatch,
        "_require_mps_environment_channels_kernel",
        capture_catalog_route,
    )
    channels = torch.randn(8, 8, 8, 8, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(8, 8, 2, 8, device="cuda", dtype=torch.complex64)
    expected = torch.einsum(
        "tbij,bipr,bjps->tbrs",
        channels,
        tensor.conj(),
        tensor,
    )
    reset_site_kernel_stats(clear_cache=True)

    actual = environment_transfer_channels(channels, tensor, compiled=compiled)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
    assert site_kernel_stats()["triton_environment_channels_calls"] == 1
    assert catalog_routes == ["FQKI-TRITON-MPS-005-A"]
    (route_event,) = site_kernel_cache_events()
    distribution = route_event["compiler_distribution"]
    assert distribution in {"triton", "flagtree"}
    assert route_event["compiler_version"]
    assert route_event["compiler_identity_status"] == "resolved"
    assert (
        route_event["integration_path"]
        == {
            "triton": "direct",
            "flagtree": "flagtree",
        }[distribution]
    )


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_environment_channels_kill_switch_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
    compiled: bool,
) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ENVIRONMENT", "0")
    channels = torch.randn(8, 8, 8, 8, device="cuda", dtype=torch.complex64)
    tensor = torch.randn(8, 8, 2, 8, device="cuda", dtype=torch.complex64)
    expected = torch.einsum(
        "tbij,bipr,bjps->tbrs",
        channels,
        tensor.conj(),
        tensor,
    )
    reset_site_kernel_stats(clear_cache=True)

    actual = environment_transfer_channels(channels, tensor, compiled=compiled)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
    assert site_kernel_stats()["triton_environment_channels_calls"] == 0
