"""Catalog dispatch contracts for the Hermitian MPS observable adjoint."""

from __future__ import annotations

import pytest
import torch

import flagquantum.runtime.executors.mps.reverse_observables as reverse_observables
import flagquantum.simulation.mps.observable_adjoint_dispatch as adjoint_dispatch
from flagquantum.runtime.executors.mps.reverse_observables import (
    mps_expectation_and_adjoints,
)
from flagquantum.runtime.executors.mps.state import RankOwnedMPSState
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.observable_adjoint_dispatch import (
    _mps_observable_adjoint_dispatch_enabled,
    _mps_observable_adjoint_kernel_enabled,
    _mps_observable_adjoint_kernel_match,
    _require_mps_observable_adjoint_kernel,
)
from flagquantum.simulation.mps.observables import mps_local_observable_adjoint
from flagquantum.simulation.mps.site_kernels import (
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)

pytestmark = pytest.mark.unit


def _hermitian(value: torch.Tensor) -> torch.Tensor:
    return value + value.mH


def _inputs(
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    device: str,
) -> tuple[torch.Tensor, ...]:
    tensor = torch.randn(
        batch, left_dim, 2, right_dim, device=device, dtype=torch.complex64
    )
    left = _hermitian(
        torch.randn(batch, left_dim, left_dim, device=device, dtype=torch.complex64)
    )
    right = _hermitian(
        torch.randn(batch, right_dim, right_dim, device=device, dtype=torch.complex64)
    )
    operator = _hermitian(torch.randn(2, 2, device=device, dtype=torch.complex64))
    weights = torch.randn(batch, device=device, dtype=torch.float32)
    return tensor, left, right, operator, weights


def _autograd_reference(args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    tensor, left, right, operator, weights = args
    variable = tensor.detach().requires_grad_(True)
    values = torch.real(
        torch.einsum(
            "bij,bipr,pq,bjqs,brs->b",
            left,
            variable.conj(),
            operator,
            variable,
            right,
        )
    )
    return torch.autograd.grad(torch.sum(weights * values), variable)[0].detach()


def test_mps_observable_adjoint_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_observable_adjoint_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert (
        implementation.semantic_id == "mps.gradient.hermitian_observable_adjoint.local"
    )
    assert implementation.implementation_id == "FQKI-TRITON-MPS-006-A"
    assert implementation.symbol == "fused_mps_hermitian_observable_adjoint"
    assert implementation.directions == ("vjp",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_observable_adjoint_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_observable_adjoint_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-006-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_observable_adjoint_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_observable_adjoint_kernel(
            device_type="cuda",
            dtype="complex128",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_mps_observable_adjoint_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    assert _mps_observable_adjoint_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", disabled)
    assert not _mps_observable_adjoint_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", "1")
    assert _mps_observable_adjoint_dispatch_enabled()


def test_mps_observable_adjoint_route_rejects_cpu_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    inputs = _inputs(2, 4, 4, device="cpu")

    assert not _mps_observable_adjoint_kernel_enabled(*inputs, hermitian=True)


def test_mps_observable_adjoint_reference_path_reports_fallback(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    inputs = _inputs(2, 4, 4, device="cpu")
    reset_site_kernel_stats(clear_cache=True)

    actual = mps_local_observable_adjoint(*inputs, hermitian=True)

    torch.testing.assert_close(actual, _autograd_reference(inputs))
    stats = site_kernel_stats()
    assert stats["triton_observable_adjoint_calls"] == 0
    assert stats["observable_adjoint_fallback_calls"] == 1
    assert site_kernel_cache_events() == ()


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_observable_adjoint_route_enforces_evidenced_window(monkeypatch) -> None:
    inputs = _inputs(8, 16, 16, device="cuda")

    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    assert _mps_observable_adjoint_kernel_enabled(*inputs, hermitian=True)

    monkeypatch.setenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", "0")
    assert not _mps_observable_adjoint_kernel_enabled(*inputs, hermitian=True)
    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    tensor, left, right, operator, weights = inputs
    assert not _mps_observable_adjoint_kernel_enabled(
        tensor,
        left.transpose(-2, -1),
        right,
        operator,
        weights,
        hermitian=True,
    )
    outside_window = _inputs(1, 64, 64, device="cuda")
    assert not _mps_observable_adjoint_kernel_enabled(
        *outside_window,
        hermitian=True,
    )


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_observable_adjoint_route_requires_hermitian_contract(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", "1")
    inputs = _inputs(2, 4, 4, device="cuda")

    assert not _mps_observable_adjoint_kernel_enabled(*inputs, hermitian=False)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_observable_adjoint_runtime_uses_catalog(monkeypatch) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", raising=False)
    monkeypatch.setattr(
        reverse_observables.dist,
        "broadcast",
        lambda tensor, *, src: None,
    )
    catalog_routes: list[str] = []
    require_cataloged_kernel = adjoint_dispatch._require_mps_observable_adjoint_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        adjoint_dispatch,
        "_require_mps_observable_adjoint_kernel",
        capture_catalog_route,
    )
    torch.manual_seed(98)
    tensor = torch.randn(8, 1, 2, 1, device="cuda", dtype=torch.complex64)
    state = RankOwnedMPSState(1, 8, 0, 1, MPSConfig(), {0: tensor}, ((0,),))
    reset_site_kernel_stats(clear_cache=True)

    value, adjoints = mps_expectation_and_adjoints(state, {0: "z"}, 0.25)

    vector = tensor.reshape(8, 2)
    operator = tensor.new_tensor([[1.0, 0.0], [0.0, -1.0]])
    values = torch.einsum("bi,ij,bj->b", vector.conj(), operator, vector).real
    expected_value = (values - 0.25).square().mean()
    variable = tensor.detach().requires_grad_(True)
    variable_vector = variable.reshape(8, 2)
    variable_values = torch.einsum(
        "bi,ij,bj->b",
        variable_vector.conj(),
        operator,
        variable_vector,
    ).real
    (expected_adjoint,) = torch.autograd.grad(
        (variable_values - 0.25).square().mean(),
        variable,
    )

    torch.testing.assert_close(value, expected_value, rtol=2e-4, atol=1e-4)
    torch.testing.assert_close(adjoints[0], expected_adjoint, rtol=1e-3, atol=5e-4)
    stats = site_kernel_stats()
    assert stats["triton_observable_adjoint_calls"] == 1
    assert stats["observable_adjoint_fallback_calls"] == 0
    assert catalog_routes == ["FQKI-TRITON-MPS-006-A"]
    (route_event,) = site_kernel_cache_events()
    assert route_event["semantic_id"] == (
        "mps.gradient.hermitian_observable_adjoint.local"
    )
    assert route_event["implementation_id"] == "FQKI-TRITON-MPS-006-A"
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
def test_mps_observable_adjoint_kill_switch_uses_reference(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", "0")
    inputs = _inputs(8, 8, 8, device="cuda")
    expected = _autograd_reference(inputs)
    reset_site_kernel_stats(clear_cache=True)

    actual = mps_local_observable_adjoint(*inputs, hermitian=True)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)
    stats = site_kernel_stats()
    assert stats["triton_observable_adjoint_calls"] == 0
    assert stats["observable_adjoint_fallback_calls"] == 1
    assert site_kernel_cache_events() == ()
