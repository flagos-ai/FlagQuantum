"""Catalog dispatch contracts for fused local RX/RZ sequences."""

from __future__ import annotations

import pytest

from flagquantum.simulation.statevector.rx_rz_dispatch import (
    _require_rx_rz_sequence_kernel,
    _rx_rz_sequence_kernel_match,
)

pytestmark = pytest.mark.unit


def test_rx_rz_sequence_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_rx_rz_sequence_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.rx_rz_sequence.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-005-A"
    assert implementation.symbol == "repeated_rx_rz"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_rx_rz_sequence_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _rx_rz_sequence_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-005-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_rx_rz_sequence_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*device"):
        _require_rx_rz_sequence_kernel(
            device_type="cpu",
            dtype="complex64",
        )
