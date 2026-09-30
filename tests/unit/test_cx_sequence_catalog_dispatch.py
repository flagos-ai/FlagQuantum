"""Catalog dispatch contracts for fused local CNOT sequences."""

from __future__ import annotations

import pytest

from flagquantum.simulation.statevector.cx_sequence_dispatch import (
    _cx_sequence_kernel_match,
    _require_cx_sequence_kernel,
)

pytestmark = pytest.mark.unit


def test_cx_sequence_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_cx_sequence_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.cnot_sequence.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-003-B"
    assert implementation.symbol == "cx_sequence"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_cx_sequence_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _cx_sequence_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-003-B"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_cx_sequence_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*device"):
        _require_cx_sequence_kernel(
            device_type="cpu",
            dtype="complex64",
        )
