"""Catalog and rollout contracts for local diagonal matrices."""

from __future__ import annotations

import sys

import pytest
import torch

import flagquantum.simulation.statevector.diagonal_matrix_dispatch as dispatch
from flagquantum.simulation.statevector.diagonal_matrix_dispatch import (
    _diagonal_matrix_dispatch_enabled,
    _diagonal_matrix_kernel_match,
    _diagonal_matrix_shape_supported,
    _require_diagonal_matrix_kernel,
)
from flagquantum.simulation.statevector.operations import _apply_diagonal_matrix

pytestmark = pytest.mark.unit


def test_diagonal_matrix_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_diagonal_matrix_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.diagonal.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-010-A"
    assert implementation.symbol == "apply_complex64_local_diagonal"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_diagonal_matrix_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _diagonal_matrix_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-010-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_diagonal_matrix_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_diagonal_matrix_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_diagonal_matrix_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_DIAGONAL_MATRIX", raising=False)
    assert _diagonal_matrix_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_DIAGONAL_MATRIX", disabled)
    assert not _diagonal_matrix_dispatch_enabled()


@pytest.mark.parametrize(
    ("state_shape", "matrix_shape", "qubits", "n_qubits", "supported"),
    (
        ((1, 1 << 16), (2, 2), (0,), 16, True),
        ((1, 1 << 24), (4, 4), (3, 19), 24, True),
        ((4, 1 << 20), (4, 4, 4), (1, 18), 20, True),
        ((1, 1 << 15), (2, 2), (0,), 15, False),
        ((2, 1 << 20), (2, 2), (0,), 20, False),
        ((1, 1 << 20), (8, 8), (0, 1, 2), 20, False),
        ((1, 1 << 20), (4, 4), (1, 1), 20, False),
        ((1, 1 << 20), (4, 4), (0, 20), 20, False),
        ((1, 1 << 19), (2, 2), (0,), 20, False),
    ),
)
def test_diagonal_matrix_shape_policy_matches_evidenced_window(
    state_shape: tuple[int, ...],
    matrix_shape: tuple[int, ...],
    qubits: tuple[int, ...],
    n_qubits: int,
    supported: bool,
) -> None:
    assert (
        _diagonal_matrix_shape_supported(
            state_shape,
            matrix_shape,
            qubits=qubits,
            n_qubits=n_qubits,
        )
        is supported
    )


def test_diagonal_matrix_cpu_preserves_reference_and_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_DIAGONAL_MATRIX", "1")
    sys.modules.pop("flagquantum.kernels.triton.statevector_diagonal", None)
    state = torch.randn(1, 1 << 4, dtype=torch.complex64)
    matrix = torch.diag(torch.randn(4, dtype=torch.complex64))

    actual = _apply_diagonal_matrix(state, matrix, (0, 3), 4)

    assert actual.shape == state.shape
    assert "flagquantum.kernels.triton.statevector_diagonal" not in sys.modules


def test_diagonal_matrix_runtime_falls_back_when_support_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.randn(1, 1 << 4, dtype=torch.complex64)
    matrix = torch.diag(torch.randn(2, dtype=torch.complex64))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unsupported inputs must not execute SV-010")

    monkeypatch.setattr(dispatch, "_apply_cataloged_diagonal_matrix", fail_if_called)

    actual = _apply_diagonal_matrix(state, matrix, (0,), 4)

    assert actual.shape == state.shape


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_diagonal_matrix_public_path_uses_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_DIAGONAL_MATRIX", "1")
    routes: list[str] = []
    require_cataloged_kernel = dispatch._require_diagonal_matrix_kernel

    def capture_catalog_route(*, device_type: str, dtype: str):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        dispatch,
        "_require_diagonal_matrix_kernel",
        capture_catalog_route,
    )
    state = torch.randn(1, 1 << 16, device="cuda", dtype=torch.complex64)
    diagonal = torch.randn(4, device="cuda", dtype=torch.complex64)
    matrix = torch.diag(diagonal)

    actual = _apply_diagonal_matrix(state, matrix, (0, 15), 16)
    expected = (
        state
        * diagonal[
            2 * ((torch.arange(1 << 16, device="cuda") >> 15) & 1)
            + (torch.arange(1 << 16, device="cuda") & 1)
        ]
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert routes == ["FQKI-TRITON-SV-010-A"]


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_diagonal_matrix_kill_switch_preserves_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_TRITON_DIAGONAL_MATRIX", "0")
    state = torch.randn(1, 1 << 16, device="cuda", dtype=torch.complex64)
    matrix = torch.diag(torch.randn(2, device="cuda", dtype=torch.complex64))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled dispatch must not execute SV-010")

    monkeypatch.setattr(dispatch, "_apply_cataloged_diagonal_matrix", fail_if_called)

    actual = _apply_diagonal_matrix(state, matrix, (3,), 16)

    assert actual.shape == state.shape
