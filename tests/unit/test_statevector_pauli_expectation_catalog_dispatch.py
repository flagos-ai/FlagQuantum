"""Catalog dispatch contracts for statevector Pauli-product expectations."""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.pauli_expectation_dispatch as dispatch
from flagquantum.simulation.statevector.pauli_expectation_dispatch import (
    _require_statevector_pauli_expectation_kernel,
    _statevector_pauli_expectation_dispatch_enabled,
    _statevector_pauli_expectation_kernel_enabled,
    _statevector_pauli_expectation_kernel_match,
)

pytestmark = pytest.mark.unit

_OPERATORS = ((0, "X"), (12, "Y"), (23, "Z"))


def test_statevector_pauli_expectation_dispatch_binds_exact_catalog_entry() -> None:
    implementation = _require_statevector_pauli_expectation_kernel(
        device_type="cuda",
        dtype="complex64",
        direction="backward",
    )

    assert (
        implementation.semantic_id
        == "measurement.expectation.pauli_product.statevector"
    )
    assert implementation.implementation_id == "FQKI-TRITON-MEAS-002-A"
    assert implementation.symbol == "statevector_pauli_expectation"
    assert implementation.directions == ("forward", "backward")
    assert implementation.maturity == "experimental"


@pytest.mark.parametrize(
    ("device_type", "dtype", "direction", "mismatch"),
    (
        ("cpu", "complex64", "forward", "device"),
        ("cuda", "complex128", "forward", "dtype"),
    ),
)
def test_statevector_pauli_expectation_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    direction: str,
    mismatch: str,
) -> None:
    match = _statevector_pauli_expectation_kernel_match(
        device_type=device_type,
        dtype=dtype,
        direction=direction,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MEAS-002-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_statevector_pauli_expectation_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_statevector_pauli_expectation_kernel(
            device_type="cuda",
            dtype="complex128",
            direction="forward",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_statevector_pauli_expectation_rollout_defaults_on_and_has_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", raising=False)
    assert _statevector_pauli_expectation_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", disabled)
    assert not _statevector_pauli_expectation_dispatch_enabled()


def test_statevector_pauli_expectation_cpu_preserves_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", "1")
    state = torch.randn(1, 8, dtype=torch.complex64, requires_grad=True)
    circuit = fq.Circuit(3, inputs=state)

    actual = circuit.expectation_ps(x=(0,), y=(1,), z=(2,))

    assert actual.shape == (1,)
    (gradient,) = torch.autograd.grad(actual.sum(), state)
    assert gradient.shape == state.shape


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_pauli_expectation_route_enforces_evidenced_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", "1")
    state = torch.empty(1, 1 << 24, device="cuda", dtype=torch.complex64)

    assert _statevector_pauli_expectation_kernel_enabled(state, _OPERATORS)
    assert not _statevector_pauli_expectation_kernel_enabled(
        state,
        ((0, "X"), (11, "Y"), (23, "Z")),
    )
    assert not _statevector_pauli_expectation_kernel_enabled(state[:, ::2], _OPERATORS)
    assert not _statevector_pauli_expectation_kernel_enabled(
        torch.empty(16, 1 << 20, device="cuda", dtype=torch.complex64),
        _OPERATORS,
    )
    assert not _statevector_pauli_expectation_kernel_enabled(
        state.to(torch.complex128),
        _OPERATORS,
    )


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_pauli_expectation_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", "1")
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_statevector_pauli_expectation_kernel

    def capture_catalog_route(*, device_type: str, dtype: str, direction: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
            direction=direction,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_statevector_pauli_expectation_kernel",
        capture_catalog_route,
    )
    state = torch.randn(1, 1 << 24, device="cuda", dtype=torch.complex64)
    state = (
        state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    ).requires_grad_()

    actual = fq.Circuit(24, inputs=state, device=state.device).expectation_ps(
        x=(0,),
        y=(12,),
        z=(23,),
    )

    assert actual.shape == (1,)
    (gradient,) = torch.autograd.grad(actual.sum(), state)
    assert gradient.shape == state.shape
    assert routes == ["FQKI-TRITON-MEAS-002-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_pauli_expectation_kill_switch_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION", "0")
    state = torch.randn(1, 1 << 24, device="cuda", dtype=torch.complex64)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("catalog kernel must not run")

    monkeypatch.setattr(
        dispatch,
        "_apply_cataloged_statevector_pauli_expectation",
        fail_if_called,
    )

    actual = fq.Circuit(24, inputs=state, device=state.device).expectation_ps(
        x=(0,),
        y=(12,),
        z=(23,),
    )

    assert actual.shape == (1,)
    assert not called
