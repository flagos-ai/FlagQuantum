from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.one_site_dispatch as one_site_dispatch
from flagquantum.simulation.mps.one_site_dispatch import (
    _apply_mps_one_site,
    _mps_one_site_kernel_enabled,
    _mps_one_site_kernel_match,
    _mps_one_site_rollout_enabled,
    _require_mps_one_site_kernel,
)
from flagquantum.simulation.mps.site_kernels import (
    apply_ry_bucket,
    reset_site_kernel_stats,
    site_kernel_stats,
)
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_mps_one_site_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_one_site_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.contract.one_site_gate"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-003-A"
    assert implementation.symbol == "fused_mps_one_site"
    assert implementation.directions == ("forward", "backward")
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_one_site_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_one_site_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-003-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_one_site_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_one_site_kernel(device_type="cuda", dtype="complex128")


def test_mps_one_site_rollout_defaults_on_and_supports_kill_switch(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ONE_SITE", raising=False)
    assert _mps_one_site_rollout_enabled()

    for value in ("0", "false", "OFF", "No"):
        monkeypatch.setenv("FQ_TRITON_MPS_ONE_SITE", value)
        assert not _mps_one_site_rollout_enabled()


def test_mps_one_site_route_rejects_cpu_bucket_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ONE_SITE", raising=False)
    tensor = torch.randn(2, 2, 64, 2, 64, dtype=torch.complex64)
    gate = torch.randn(2, 2, 2, 2, dtype=torch.complex64)

    assert not _mps_one_site_kernel_enabled(
        tensor.reshape(4, 64, 2, 64), gate.reshape(4, 2, 2)
    )


def test_mps_summary_reports_default_and_disabled_rollout(monkeypatch) -> None:
    state = MPSState((torch.tensor([[[[1.0 + 0.0j], [0.0j]]]], dtype=torch.complex64),))
    monkeypatch.delenv("FQ_TRITON_MPS_ONE_SITE", raising=False)
    assert state.summary()["triton_mps_one_site_enabled"] is True

    monkeypatch.setenv("FQ_TRITON_MPS_ONE_SITE", "0")
    assert state.summary()["triton_mps_one_site_enabled"] is False


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_one_site_kill_switch_uses_reference(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ONE_SITE", "0")
    tensor = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.complex64)
    gate = torch.randn(2, 2, device="cuda", dtype=torch.complex64)
    expected = torch.einsum("pq,blqr->blpr", gate, tensor)

    actual, routed = _apply_mps_one_site(tensor, gate)

    assert not routed
    assert torch.allclose(actual, expected)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_state_apply_one_uses_catalog_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ONE_SITE", raising=False)
    catalog_routes = []
    require_cataloged_kernel = one_site_dispatch._require_mps_one_site_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        one_site_dispatch,
        "_require_mps_one_site_kernel",
        capture_catalog_route,
    )
    tensors = (
        torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.complex64),
        torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.complex64),
        torch.randn(1, 64, 2, 1, device="cuda", dtype=torch.complex64),
    )
    state = MPSState(tensors)
    gate = torch.randn(2, 2, device="cuda", dtype=torch.complex64)
    expected = torch.einsum("pq,blqr->blpr", gate, tensors[1])

    state.apply_one(gate, 1)

    assert torch.allclose(state.tensors[1], expected, rtol=2e-5, atol=2e-5)
    assert state.triton_one_site_regions == 1
    assert catalog_routes == ["FQKI-TRITON-MPS-003-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_one_site_bucket_uses_catalog_by_default_and_preserves_gradients(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_ONE_SITE", raising=False)
    catalog_routes = []
    require_cataloged_kernel = one_site_dispatch._require_mps_one_site_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        one_site_dispatch,
        "_require_mps_one_site_kernel",
        capture_catalog_route,
    )
    tensors = torch.randn(
        4,
        2,
        32,
        2,
        32,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=True,
    )
    angles = torch.randn(4, 2, device="cuda")
    gates = torch.zeros(4, 2, 2, 2, device="cuda", dtype=torch.complex64)
    gates[..., 0, 0] = torch.cos(angles / 2)
    gates[..., 0, 1] = -torch.sin(angles / 2)
    gates[..., 1, 0] = torch.sin(angles / 2)
    gates[..., 1, 1] = torch.cos(angles / 2)
    gates.requires_grad_(True)
    reference_tensors = tensors.detach().clone().requires_grad_(True)
    reference_gates = gates.detach().clone().requires_grad_(True)
    cotangent = torch.randn_like(tensors)
    expected = torch.einsum("kbpq,kblqr->kblpr", reference_gates, reference_tensors)
    reset_site_kernel_stats(clear_cache=True)

    actual = apply_ry_bucket(tensors, gates, compiled=True)
    actual.backward(cotangent)
    expected.backward(cotangent)

    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.allclose(tensors.grad, reference_tensors.grad, rtol=2e-5, atol=2e-5)
    assert torch.allclose(gates.grad, reference_gates.grad, rtol=2e-5, atol=2e-5)
    assert site_kernel_stats()["triton_one_site_bucket_calls"] == 1
    assert catalog_routes == ["FQKI-TRITON-MPS-003-A"]
