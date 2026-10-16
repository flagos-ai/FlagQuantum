"""Catalog and rollout contracts for local dense two-qubit matrices."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum.simulation.statevector.two_qubit_matrix_dispatch as dispatch
from flagquantum.simulation.statevector.operations import _apply_matrix
from flagquantum.simulation.statevector.two_qubit_matrix_dispatch import (
    _require_two_qubit_matrix_kernel,
    _two_qubit_matrix_dispatch_enabled,
    _two_qubit_matrix_kernel_match,
    _two_qubit_matrix_shape_supported,
)

pytestmark = pytest.mark.unit


def test_two_qubit_matrix_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_two_qubit_matrix_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.matrix_2q.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-009-A"
    assert implementation.symbol == "apply_complex64_local_2q"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_two_qubit_matrix_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _two_qubit_matrix_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-009-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_two_qubit_matrix_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_two_qubit_matrix_kernel(
            device_type="cuda",
            dtype="complex128",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_two_qubit_matrix_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_TWO_QUBIT_MATRIX", raising=False)
    assert _two_qubit_matrix_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_TWO_QUBIT_MATRIX", disabled)
    assert not _two_qubit_matrix_dispatch_enabled()


@pytest.mark.parametrize(
    ("state_shape", "matrix_shape", "wires", "n_wires", "supported"),
    (
        ((1, 1 << 16), (4, 4), (0, 15), 16, True),
        ((4, 1 << 20), (4, 4), (10, 3), 20, True),
        ((1, 1 << 15), (4, 4), (0, 1), 15, False),
        ((2, 1 << 20), (4, 4), (0, 1), 20, False),
        ((1, 1 << 20), (2, 2), (0, 1), 20, False),
        ((1, 1 << 20), (4, 4), (1, 1), 20, False),
        ((1, 1 << 20), (4, 4), (0, 20), 20, False),
        ((1, 1 << 19), (4, 4), (0, 1), 20, False),
    ),
)
def test_two_qubit_matrix_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    matrix_shape: tuple[int, ...],
    wires: tuple[int, ...],
    n_wires: int,
    supported: bool,
) -> None:
    assert (
        _two_qubit_matrix_shape_supported(
            state_shape,
            matrix_shape,
            qubits=wires,
            n_qubits=n_wires,
        )
        is supported
    )


def test_two_qubit_matrix_cpu_runtime_preserves_reference_and_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_TWO_QUBIT_MATRIX", "1")
    sys.modules.pop("flagquantum.kernels.triton.statevector_gates", None)
    state = torch.randn(1, 1 << 4, dtype=torch.complex64)
    matrix = torch.randn(4, 4, dtype=torch.complex64)

    actual = _apply_matrix(state, matrix, (0, 3), 4)

    assert actual.shape == state.shape
    assert "flagquantum.kernels.triton.statevector_gates" not in sys.modules


def test_two_qubit_matrix_runtime_falls_back_when_support_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.randn(1, 1 << 4, dtype=torch.complex64)
    matrix = torch.randn(4, 4, dtype=torch.complex64)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unsupported inputs must not execute SV-009")

    monkeypatch.setattr(dispatch, "_apply_cataloged_two_qubit_matrix", fail_if_called)

    actual = _apply_matrix(state, matrix, (0, 3), 4)

    assert actual.shape == state.shape
