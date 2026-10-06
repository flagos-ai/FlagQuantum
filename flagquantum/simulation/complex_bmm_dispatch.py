"""Catalog authorization for layout-aware complex batched matrix products."""

from __future__ import annotations

import torch

from ..kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from .kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-NUM-002-A"


def _layout_complex_bmm_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the layout-aware BMM kernel against its exact catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="numerics.matmul.complex_batched_layout",
            device=device_type,
            dtype=dtype,
            layout="explicit_strided_batch",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_layout_complex_bmm_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _layout_complex_bmm_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="layout-aware complex BMM kernel",
    )


def _apply_cataloged_layout_complex_bmm(
    left: torch.Tensor,
    right: torch.Tensor,
    left_permutation: tuple[int, ...],
    right_permutation: tuple[int, ...],
    shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    """Execute the layout-aware BMM after exact catalog authorization."""

    _require_layout_complex_bmm_kernel(
        device_type=left.device.type,
        dtype=str(left.dtype).removeprefix("torch."),
    )
    from ..kernels.triton.complex_bmm import fused_complex_layout_bmm

    return fused_complex_layout_bmm(
        left,
        right,
        left_permutation,
        right_permutation,
        shapes,
    )


__all__ = (
    "_apply_cataloged_layout_complex_bmm",
    "_layout_complex_bmm_kernel_match",
    "_require_layout_complex_bmm_kernel",
)
