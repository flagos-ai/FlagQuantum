"""Catalog authorization for fused MPS two-site contractions."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-001-A"


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


def _require_mps_two_site_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _mps_two_site_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
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


__all__ = (
    "_apply_cataloged_mps_two_site",
    "_mps_two_site_kernel_match",
    "_require_mps_two_site_kernel",
)
