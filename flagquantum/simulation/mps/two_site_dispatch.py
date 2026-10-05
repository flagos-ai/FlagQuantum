"""Catalog authorization for fused MPS two-site contractions."""

from __future__ import annotations

from functools import lru_cache

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_TWO_SITE_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-001-A"
_PROJECTED_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-002-A"


def _mps_two_site_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the fused two-site contraction against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.contract.two_site_gate",
            device=device_type,
            dtype=dtype,
            layout="mps_two_site",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_mps_two_site_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _mps_two_site_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_TWO_SITE_IMPLEMENTATION_ID,
        description="fused MPS two-site kernel",
    )


def _apply_cataloged_mps_two_site(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Execute the fused contraction after exact catalog authorization."""

    _require_mps_two_site_kernel(
        device_type=left.device.type,
        dtype=str(left.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_two_site import fused_mps_two_site

    return fused_mps_two_site(left, gate, right)


def _mps_projected_two_site_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the projected two-site contraction against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.contract.two_site_gate_projected",
            device=device_type,
            dtype=dtype,
            layout="projected_range",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_mps_projected_two_site_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the projected implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _mps_projected_two_site_kernel_match(
            device_type=device_type,
            dtype=dtype,
        ),
        implementation_id=_PROJECTED_IMPLEMENTATION_ID,
        description="projected MPS two-site kernel",
    )


def _apply_cataloged_mps_projected_two_site(
    left: torch.Tensor,
    gate: torch.Tensor,
    right: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Execute the projected contraction after catalog authorization."""

    _require_mps_projected_two_site_kernel(
        device_type=left.device.type,
        dtype=str(left.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_two_site import fused_mps_range_projection

    return fused_mps_range_projection(left, gate, right, projection)


__all__ = (
    "_apply_cataloged_mps_projected_two_site",
    "_apply_cataloged_mps_two_site",
    "_mps_projected_two_site_kernel_match",
    "_mps_two_site_kernel_match",
    "_require_mps_projected_two_site_kernel",
    "_require_mps_two_site_kernel",
)
