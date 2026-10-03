"""Catalog dispatch contracts for statevector marginal probabilities."""

from __future__ import annotations

import pytest
import torch

import flagquantum.runtime.statevector_measurement_dispatch as dispatch
from flagquantum.runtime.measurements import _joint_marginal_probabilities
from flagquantum.runtime.statevector_measurement_dispatch import (
    _require_statevector_marginal_kernel,
    _statevector_marginal_dispatch_enabled,
    _statevector_marginal_kernel_enabled,
    _statevector_marginal_kernel_match,
)

pytestmark = pytest.mark.unit


def test_statevector_marginal_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_statevector_marginal_kernel(
        device_type="cuda",
        dtype="complex64",
        direction="backward",
    )

    assert implementation.semantic_id == (
        "measurement.probabilities.marginal.statevector"
    )
    assert implementation.implementation_id == "FQKI-TRITON-MEAS-003-A"
    assert implementation.symbol == "statevector_marginal_probabilities"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "direction", "mismatch"),
    (
        ("cpu", "complex64", "forward", "device"),
        ("cuda", "complex128", "forward", "dtype"),
    ),
)
def test_statevector_marginal_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    direction: str,
    mismatch: str,
) -> None:
    match = _statevector_marginal_kernel_match(
        device_type=device_type,
        dtype=dtype,
        direction=direction,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MEAS-003-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_statevector_marginal_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_statevector_marginal_kernel(
            device_type="cuda",
            dtype="complex128",
            direction="forward",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_statevector_marginal_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv(
        "FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES",
        raising=False,
    )
    assert _statevector_marginal_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES", disabled)
    assert not _statevector_marginal_dispatch_enabled()


def test_statevector_marginal_cpu_runtime_preserves_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES", "1")
    state = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.complex64)

    actual = _joint_marginal_probabilities(
        state,
        (1,),
        n_wires=2,
        noise_model=None,
    )

    assert actual is not None
    torch.testing.assert_close(actual, torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]))


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_marginal_route_enforces_evidenced_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES", "1")
    state = torch.empty(1, 1 << 24, device="cuda", dtype=torch.complex64)

    assert _statevector_marginal_kernel_enabled(state, (23, 0, 15, 8))
    assert not _statevector_marginal_kernel_enabled(state[:, ::2], (22, 0, 8, 4))
    small = torch.empty(16, 1 << 16, device="cuda", dtype=torch.complex64)
    assert not _statevector_marginal_kernel_enabled(small, (15, 0, 7, 4))
    assert not _statevector_marginal_kernel_enabled(state, tuple(range(9)))


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_marginal_runtime_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES", "1")
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_statevector_marginal_kernel

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
        "_require_statevector_marginal_kernel",
        capture_catalog_route,
    )
    state = torch.randn(1, 1 << 24, device="cuda", dtype=torch.complex64)
    state = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)

    actual = _joint_marginal_probabilities(
        state,
        (23, 0, 15, 8),
        n_wires=24,
        noise_model=None,
    )
    probabilities = torch.abs(state) ** 2
    selected = {23, 0, 15, 8}
    axes = tuple(qubit + 1 for qubit in range(24) if qubit not in selected)
    expected = probabilities.reshape(1, *(2,) * 24).sum(dim=axes)
    expected = expected.permute(0, 4, 1, 3, 2).reshape(1, 16)

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert routes == ["FQKI-TRITON-MEAS-003-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statevector_marginal_kill_switch_uses_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES", "0")
    state = torch.randn(1, 1 << 24, device="cuda", dtype=torch.complex64)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("catalog kernel must not run")

    monkeypatch.setattr(
        dispatch,
        "_apply_cataloged_statevector_marginal",
        fail_if_called,
    )

    actual = _joint_marginal_probabilities(
        state,
        (23, 0, 15, 8),
        n_wires=24,
        noise_model=None,
    )

    assert actual is not None
    assert not called
