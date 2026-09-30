from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.one_site_dispatch as one_site_dispatch
from flagquantum.simulation.mps.one_site_dispatch import (
    _mps_one_site_kernel_match,
    _require_mps_one_site_kernel,
)
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_mps_one_site_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_one_site_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.contract.one_site_gate"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-003-A"
    assert implementation.symbol == "fused_mps_one_site"
    assert implementation.directions == ("forward", "backward")


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_one_site_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_one_site_kernel_match(device_type=device_type, dtype=dtype)

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-003-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_one_site_dispatch_fails_closed_on_unsupported_input() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_one_site_kernel(device_type="cuda", dtype="complex128")


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_state_apply_one_uses_catalog(monkeypatch) -> None:
    monkeypatch.setenv("FQ_TRITON_MPS_ONE_SITE", "1")
    catalog_routes = []
    require_cataloged_kernel = one_site_dispatch._require_mps_one_site_kernel

    def capture_catalog_route(*, device_type, dtype):
        implementation = require_cataloged_kernel(
            device_type=device_type,
            dtype=dtype,
        )
        catalog_routes.append(implementation.implementation_id)
        return implementation

    monkeypatch.setattr(
        one_site_dispatch,
        "_require_mps_one_site_kernel",
        capture_catalog_route,
    )
    tensors = (
        torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.complex64),
        torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.complex64),
        torch.randn(1, 64, 2, 1, device="cuda", dtype=torch.complex64),
    )
    state = MPSState(tensors)
    gate = torch.randn(2, 2, device="cuda", dtype=torch.complex64)
    expected = torch.einsum("pq,blqr->blpr", gate, tensors[1])

    state.apply_one(gate, 1)

    assert torch.allclose(state.tensors[1], expected, rtol=2e-5, atol=2e-5)
    assert state.triton_one_site_regions == 1
    assert catalog_routes == ["FQKI-TRITON-MPS-003-A"]
