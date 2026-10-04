"""Catalog dispatch contracts for MPS wire-probability reductions."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.wire_probability_dispatch as probability_dispatch
from flagquantum.simulation.mps.site_kernels import (
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)
from flagquantum.simulation.mps.state import MPSState
from flagquantum.simulation.mps.wire_probability_dispatch import (
    _mps_wire_probability_kernel_enabled,
    _mps_wire_probability_kernel_match,
    _require_mps_wire_probability_kernel,
)

pytestmark = pytest.mark.unit


def test_mps_wire_probability_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_wire_probability_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.measurement.wire_probabilities.local"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-007-A"
    assert implementation.symbol == "fused_mps_qubit_probabilities"
    assert implementation.directions == ("forward",)


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_wire_probability_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_wire_probability_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-007-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_wire_probability_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_wire_probability_kernel(
            device_type="cuda",
            dtype="complex128",
        )


def test_mps_wire_probability_reference_path_reports_fallback(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", raising=False)
    state = MPSState.zero(1, bsz=3, dtype=torch.complex64)
    reset_site_kernel_stats(clear_cache=True)

    actual = state._qubit_probabilities(0)

    torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0]]).expand(3, -1))
    stats = site_kernel_stats()
    assert stats["triton_qubit_probability_calls"] == 0
    assert stats["qubit_probability_fallback_calls"] == 1
    assert site_kernel_cache_events() == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_wire_probability_route_enforces_evidenced_window(monkeypatch) -> None:
    tensor = torch.randn(8, 16, 2, 16, device="cuda", dtype=torch.complex64)

    monkeypatch.delenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", raising=False)
    assert not _mps_wire_probability_kernel_enabled(tensor)

    monkeypatch.setenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", "1")
    assert _mps_wire_probability_kernel_enabled(tensor)
    assert not _mps_wire_probability_kernel_enabled(tensor.transpose(1, 3))
    assert not _mps_wire_probability_kernel_enabled(tensor.requires_grad_(True))
    large = torch.randn(1, 65, 2, 64, device="cuda", dtype=torch.complex64)
    assert not _mps_wire_probability_kernel_enabled(large)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_wire_probability_runtime_uses_catalog(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", "1")
    catalog_routes: list[str] = []
    require_cataloged_kernel = probability_dispatch._require_mps_wire_probability_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        probability_dispatch,
        "_require_mps_wire_probability_kernel",
        capture_catalog_route,
    )
    torch.manual_seed(109)
    tensor = torch.randn(8, 1, 2, 1, device="cuda", dtype=torch.complex64)
    tensor = tensor / torch.linalg.vector_norm(tensor, dim=(1, 2, 3), keepdim=True)
    state = MPSState([tensor])
    reset_site_kernel_stats(clear_cache=True)

    actual = state._qubit_probabilities(0)
    expected = torch.sum(torch.abs(tensor) ** 2, dim=(1, 3))

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    stats = site_kernel_stats()
    assert stats["triton_wire_probability_calls"] == 1
    assert stats["wire_probability_fallback_calls"] == 0
    assert catalog_routes == ["FQKI-TRITON-MPS-007-A"]
    (route_event,) = site_kernel_cache_events()
    assert route_event["semantic_id"] == "mps.measurement.wire_probabilities.local"
    assert route_event["implementation_id"] == "FQKI-TRITON-MPS-007-A"
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


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_public_mps_sampling_routes_each_wire(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", "1")
    state = MPSState.zero(3, bsz=2, device="cuda", dtype=torch.complex64)
    reset_site_kernel_stats(clear_cache=True)

    samples = state.sample(4, generator=torch.Generator(device="cuda").manual_seed(7))

    assert torch.equal(samples, torch.zeros_like(samples))
    stats = site_kernel_stats()
    assert stats["triton_wire_probability_calls"] == 3
    assert stats["wire_probability_fallback_calls"] == 0
