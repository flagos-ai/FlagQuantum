"""Catalog authorization for layout-aware complex batched matrix products."""

from __future__ import annotations

import torch

from ..kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from .kernel_dispatch import (
    _catalog_declares_kernel,
    _operand_kernel_axis,
    _require_cataloged_kernel,
)

_IMPLEMENTATION_ID = "FQKI-TRITON-NUM-002-A"
_SEMANTIC_ID = "numerics.matmul.complex_batched_layout"
_LAYOUT = "explicit_strided_batch"


def _layout_complex_bmm_kernel_match(
    *,
    device_type: str,
    dtype: str,
) -> KernelMatchResult:
    """Match the layout-aware BMM semantic against its wired catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id=_SEMANTIC_ID,
            device=device_type,
            dtype=dtype,
            layout=_LAYOUT,
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_layout_complex_bmm_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _layout_complex_bmm_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="layout-aware complex BMM kernel",
    )


def _layout_complex_bmm_declared(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether the catalog declares a layout BMM for these operands.

    The catalog is the authority for which execution devices can run the fused
    route; the caller must not infer it from a device literal. Two operands that
    do not share one declared device and precision have no single catalog
    request to make, and are not declared here.
    """

    axis = _operand_kernel_axis((left, right))
    if axis is None:
        return False
    device_type, dtype = axis
    return _layout_complex_bmm_declared_for(device_type=device_type, dtype=dtype)


def _layout_complex_bmm_declared_for(
    *, device_type: str, dtype: str | torch.dtype
) -> bool:
    """Answer the same route question from a device and precision pair.

    Planning code that holds a requested device and dtype rather than operands
    asks the catalog the same question through this entry, so one semantic has
    one declared-route spelling across the tensor-network and MPS paths. A
    device outside the declared axis is not declared, never an error.
    """

    return _catalog_declares_kernel(
        _SEMANTIC_ID,
        device_type=device_type,
        dtype=str(dtype).removeprefix("torch."),
        layout=_LAYOUT,
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
    "_layout_complex_bmm_declared",
    "_layout_complex_bmm_declared_for",
    "_layout_complex_bmm_kernel_match",
    "_require_layout_complex_bmm_kernel",
)
