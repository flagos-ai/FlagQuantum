"""Catalog dispatch contracts for layout-aware complex BMM."""

from __future__ import annotations

import pytest

from flagquantum.simulation.complex_bmm_dispatch import (
    _layout_complex_bmm_kernel_match,
    _require_layout_complex_bmm_kernel,
)

pytestmark = pytest.mark.unit


def test_layout_complex_bmm_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_layout_complex_bmm_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "numerics.matmul.complex_batched_layout"
    assert implementation.implementation_id == "FQKI-TRITON-NUM-002-A"
    assert implementation.symbol == "fused_complex_layout_bmm"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_layout_complex_bmm_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _layout_complex_bmm_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-NUM-002-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_layout_complex_bmm_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*device"):
        _require_layout_complex_bmm_kernel(
            device_type="cpu",
            dtype="complex64",
        )
