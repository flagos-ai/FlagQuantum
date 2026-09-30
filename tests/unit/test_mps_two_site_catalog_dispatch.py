"""Catalog dispatch contracts for fused MPS two-site contractions."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.two_site_dispatch as two_site_dispatch
from flagquantum.simulation.mps.state import MPSState
from flagquantum.simulation.mps.two_site_dispatch import (
    _mps_two_site_kernel_match,
    _require_mps_two_site_kernel,
)

pytestmark = pytest.mark.unit


def test_mps_two_site_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_two_site_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.contract.two_site_gate"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-001-A"
    assert implementation.symbol == "fused_mps_two_site"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_two_site_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_two_site_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-001-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_two_site_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_two_site_kernel(
            device_type="cuda",
            dtype="complex128",
        )


def _interior_bond_state(n_wires: int, bond_dim: int) -> MPSState:
    bond_dims = (1, *((bond_dim,) * (n_wires - 1)), 1)
    tensors = [
        torch.randn(
            1,
            bond_dims[wire],
            2,
            bond_dims[wire + 1],
            device="cuda",
            dtype=torch.complex64,
        )
        for wire in range(n_wires)
    ]
    return MPSState(tensors)


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_state_single_and_bucket_paths_use_catalog(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_TWO_SITE", "1")
    catalog_routes = []
    require_cataloged_kernel = two_site_dispatch._require_mps_two_site_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        two_site_dispatch,
        "_require_mps_two_site_kernel",
        capture_catalog_route,
    )
    gate = torch.randn(4, 4, device="cuda", dtype=torch.complex64)
    single = _interior_bond_state(4, 64)
    bucket = _interior_bond_state(6, 64)

    single.apply_two(gate, 1)
    bucket.apply_two_bucket((gate, gate), (1, 3))

    assert single.triton_two_site_regions == 1
    assert bucket.triton_two_site_regions == 2
    assert catalog_routes == ["FQKI-TRITON-MPS-001-A"] * 2
