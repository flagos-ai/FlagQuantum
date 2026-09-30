"""Catalog dispatch contracts for fused local RY/RZ pairs."""

from __future__ import annotations

import pytest

from flagquantum.simulation.statevector.ry_rz_dispatch import (
    _require_ry_rz_pair_kernel,
    _ry_rz_pair_kernel_match,
)

pytestmark = pytest.mark.unit


def test_ry_rz_pair_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_ry_rz_pair_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "statevector.apply.ry_rz_pair.local"
    assert implementation.implementation_id == "FQKI-TRITON-SV-004-A"
    assert implementation.symbol == "ry_rz_pair"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_ry_rz_pair_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _ry_rz_pair_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-SV-004-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_ry_rz_pair_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_ry_rz_pair_kernel(
            device_type="cuda",
            dtype="complex128",
        )
