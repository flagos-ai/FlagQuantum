"""Catalog dispatch contracts for local single-qubit matrices."""

from __future__ import annotations

import pytest

from flagquantum.simulation.statevector.single_qubit_matrix_dispatch import (
    _require_single_qubit_matrix_kernel,
    _single_qubit_matrix_kernel_match,
)

pytestmark = pytest.mark.unit


def test_single_qubit_matrix_dispatch_selects_differentiable_implementation() -> None:
    match = _single_qubit_matrix_kernel_match(
        device_type="cuda",
        dtype="complex64",
    )
    candidate_ids = {item.implementation.implementation_id for item in match.candidates}
    implementation = _require_single_qubit_matrix_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert "FQKI-TRITON-SV-001-A" in candidate_ids
    assert implementation.semantic_id == "statevector.apply.matrix_1q.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-001-B"
    assert implementation.symbol == "single_qubit_matrix"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_single_qubit_matrix_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _single_qubit_matrix_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-001-B"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_single_qubit_matrix_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_single_qubit_matrix_kernel(
            device_type="cuda",
            dtype="complex128",
        )
