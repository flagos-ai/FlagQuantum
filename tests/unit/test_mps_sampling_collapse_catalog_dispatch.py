"""Catalog dispatch contracts for sampled-wire MPS collapse updates."""

from __future__ import annotations

import pytest
import torch

import flagquantum.simulation.mps.sampling_collapse_dispatch as collapse_dispatch
from flagquantum.simulation.mps.sampling import _collapse_sampled_wire
from flagquantum.simulation.mps.sampling_collapse_dispatch import (
    _mps_sampling_collapse_dispatch_enabled,
    _mps_sampling_collapse_kernel_enabled,
    _mps_sampling_collapse_kernel_match,
    _require_mps_sampling_collapse_kernel,
)
from flagquantum.simulation.mps.site_kernels import (
    reset_site_kernel_stats,
    site_kernel_cache_events,
    site_kernel_stats,
)
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_mps_sampling_collapse_dispatch_binds_exact_catalog_implementation() -> None:
    implementation = _require_mps_sampling_collapse_kernel(
        device_type="cuda",
        dtype="complex64",
    )

    assert implementation.semantic_id == "mps.sampling.collapse_wire.local"
    assert implementation.implementation_id == "FQKI-TRITON-MPS-008-A"
    assert implementation.symbol == "fused_mps_sampling_collapse"
    assert implementation.directions == ("forward",)
    assert implementation.maturity == "provisional"


@pytest.mark.parametrize(
    ("device_type", "dtype", "mismatch"),
    (("cpu", "complex64", "device"), ("cuda", "complex128", "dtype")),
)
def test_mps_sampling_collapse_dispatch_reports_catalog_mismatch(
    device_type: str,
    dtype: str,
    mismatch: str,
) -> None:
    match = _mps_sampling_collapse_kernel_match(
        device_type=device_type,
        dtype=dtype,
    )

    rejection = next(
        item
        for item in match.rejections
        if item.implementation.implementation_id == "FQKI-TRITON-MPS-008-A"
    )
    assert tuple(item.code for item in rejection.mismatches) == (mismatch,)


def test_mps_sampling_collapse_dispatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not authorized.*dtype"):
        _require_mps_sampling_collapse_kernel(
            device_type="cuda",
            dtype="complex128",
        )


@pytest.mark.parametrize("disabled", ("0", "false", "off", "no", " FALSE "))
def test_mps_sampling_collapse_rollout_defaults_on_and_supports_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", raising=False)
    assert _mps_sampling_collapse_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", disabled)
    assert not _mps_sampling_collapse_dispatch_enabled()

    monkeypatch.setenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", "1")
    assert _mps_sampling_collapse_dispatch_enabled()


def test_mps_sampling_collapse_reference_path_reports_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", raising=False)
    site = torch.ones(2, 1, 2, 1, dtype=torch.complex64)
    next_site = torch.ones(2, 1, 2, 1, dtype=torch.complex64)
    bits = torch.zeros(2, dtype=torch.int64)
    reset_site_kernel_stats(clear_cache=True)

    assert (
        collapse_dispatch._try_apply_cataloged_mps_sampling_collapse(
            site,
            next_site,
            bits,
        )
        is None
    )

    stats = site_kernel_stats()
    assert stats["triton_sampling_collapse_calls"] == 0
    assert stats["sampling_collapse_fallback_calls"] == 1
    assert site_kernel_cache_events() == ()


def test_runtime_applies_catalog_result_to_both_mps_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = MPSState.zero(2, bsz=3, dtype=torch.complex64)
    bits = torch.tensor([0, 1, 0])
    collapsed = torch.randn(3, 1, 2, 1, dtype=torch.complex64)
    propagated = torch.randn(3, 1, 2, 1, dtype=torch.complex64)
    monkeypatch.setattr(
        collapse_dispatch,
        "_try_apply_cataloged_mps_sampling_collapse",
        lambda *args: (collapsed, propagated),
    )

    _collapse_sampled_wire(state, 0, bits)

    assert state.tensors[0] is collapsed
    assert state.tensors[1] is propagated
    assert state.orthogonality_center == 1
    assert state._canonical_center_valid


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mps_sampling_collapse_route_enforces_evidenced_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = torch.randn(8, 1, 2, 32, device="cuda", dtype=torch.complex64)
    next_site = torch.randn(8, 32, 2, 64, device="cuda", dtype=torch.complex64)
    bits = torch.randint(0, 2, (8,), device="cuda")

    monkeypatch.delenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", raising=False)
    assert _mps_sampling_collapse_kernel_enabled(site, next_site, bits)

    monkeypatch.setenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", "0")
    assert not _mps_sampling_collapse_kernel_enabled(site, next_site, bits)
    monkeypatch.delenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", raising=False)
    assert not _mps_sampling_collapse_kernel_enabled(
        site,
        next_site.transpose(1, 3),
        bits,
    )
    assert not _mps_sampling_collapse_kernel_enabled(
        site.requires_grad_(True),
        next_site,
        bits,
    )


@pytest.mark.gpu
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_public_mps_sampling_routes_each_nonterminal_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", raising=False)
    state = MPSState.zero(3, bsz=2, device="cuda", dtype=torch.complex64)
    reset_site_kernel_stats(clear_cache=True)

    samples = state.sample(4, generator=torch.Generator(device="cuda").manual_seed(7))

    assert torch.equal(samples, torch.zeros_like(samples))
    stats = site_kernel_stats()
    assert stats["triton_sampling_collapse_calls"] == 2
    assert stats["sampling_collapse_fallback_calls"] == 0
    route_event = next(
        event
        for event in site_kernel_cache_events()
        if event.get("semantic_id") == "mps.sampling.collapse_wire.local"
    )
    assert route_event["semantic_id"] == "mps.sampling.collapse_wire.local"
    assert route_event["implementation_id"] == "FQKI-TRITON-MPS-008-A"
