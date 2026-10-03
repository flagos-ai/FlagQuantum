"""Catalog authorization for layout-aware complex batched matrix products."""

from __future__ import annotations

import torch

from ..kernels.catalog import (
    KERNEL_DEVICES,
    KernelImplementation,
    KernelMatchResult,
    KernelProvider,
    KernelRequest,
    match_kernel_implementations,
)
from .kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-NUM-002-A"
_SEMANTIC_ID = "numerics.matmul.complex_batched_layout"
_LAYOUT = "explicit_strided_batch"


def _layout_complex_bmm_kernel_match(
    *,
    device_type: str,
    dtype: str,
    providers: tuple[KernelProvider, ...] = ("triton",),
) -> KernelMatchResult:
    """Match the layout-aware BMM semantic against its exact catalog contract.

    An empty ``providers`` filter asks the catalog a different question: not
    "is the wired provider available" but "does any evidenced implementation
    cover this device and precision". Callers that only need to know whether a
    cataloged route exists pass no provider, so a record added for another
    provider changes their answer without editing them.
    """

    return match_kernel_implementations(
        KernelRequest(
            semantic_id=_SEMANTIC_ID,
            device=device_type,
            dtype=dtype,
            layout=_LAYOUT,
            direction="forward",
            addressing=("local",),
            providers=providers,
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
    """Return whether the catalog declares a layout BMM for this device/dtype.

    The catalog is the authority for which execution devices can run the fused
    route; the caller must not infer it from a device literal. A device outside
    the declared axis cannot reach a cataloged implementation, so it is not
    declared here.
    """

    operands = (left, right)
    if len({operand.device for operand in operands}) != 1:
        return False
    device_type = left.device.type
    if device_type not in KERNEL_DEVICES:
        return False
    dtypes = {str(operand.dtype).removeprefix("torch.") for operand in operands}
    if len(dtypes) != 1:
        return False
    return _layout_complex_bmm_kernel_match(
        device_type=device_type, dtype=dtypes.pop(), providers=()
    ).matched


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
    "_layout_complex_bmm_kernel_match",
    "_require_layout_complex_bmm_kernel",
)
