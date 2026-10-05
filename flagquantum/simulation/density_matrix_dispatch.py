"""Catalog authorization for the density-matrix batched matrix product.

The density-matrix simulator contracts a batched operator with a batched
density matrix. That product is the ``numerics.matmul.complex_batched``
semantic, and this module is its first wired consumer: the route question is
asked once here, from the operands' own declared device and precision, so the
simulator never spells an execution device itself.
"""

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

_IMPLEMENTATION_ID = "FQKI-TRITON-NUM-001-A"
_SEMANTIC_ID = "numerics.matmul.complex_batched"
_LAYOUT = "batched_matrix"

_STATS = {"cataloged_routes": 0, "reference_routes": 0}


def density_matmul_route_stats() -> dict[str, int]:
    """Return how many dense density-matrix products took each route."""

    return dict(_STATS)


def reset_density_matmul_route_stats() -> None:
    """Reset the route counters so a focused test can observe one call."""

    _STATS["cataloged_routes"] = 0
    _STATS["reference_routes"] = 0


def _density_matmul_kernel_match(
    *,
    device_type: str,
    dtype: str,
) -> KernelMatchResult:
    """Match the batched complex matmul semantic against its wired contract."""

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


def _require_density_matmul_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _density_matmul_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="batched complex matrix product kernel",
    )


def _density_matmul_declared(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether the catalog declares a batched complex matmul here.

    The declared ``batched_matrix`` layout is a dense ``(batch, rows, columns)``
    operand pair, so operands of another rank are not that layout and are not
    declared here. The catalog, not a device literal, decides the rest.
    """

    if left.ndim != 3 or right.ndim != 3:
        return False
    axis = _operand_kernel_axis((left, right))
    if axis is None:
        return False
    device_type, dtype = axis
    return _catalog_declares_kernel(
        _SEMANTIC_ID, device_type=device_type, dtype=dtype, layout=_LAYOUT
    )


def _apply_cataloged_density_matmul(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Execute the batched complex matmul after exact catalog authorization."""

    _require_density_matmul_kernel(
        device_type=left.device.type,
        dtype=str(left.dtype).removeprefix("torch."),
    )
    from ..kernels.triton.complex_bmm import fused_complex_bmm

    return fused_complex_bmm(left, right)


def _density_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Contract two batched complex matrices through the authorized route.

    Where the catalog declares no fused implementation for these operands,
    ``torch.bmm`` is the reference route: it is the semantic authority and the
    test oracle, and it must stay runnable with no accelerator toolchain
    installed. Every product is counted, so which route ran is observable.
    """

    if _density_matmul_declared(left, right):
        result = _apply_cataloged_density_matmul(left, right)
        _STATS["cataloged_routes"] += 1
        return result
    _STATS["reference_routes"] += 1
    return torch.bmm(left, right)


__all__ = (
    "_apply_cataloged_density_matmul",
    "_density_bmm",
    "_density_matmul_declared",
    "_density_matmul_kernel_match",
    "_require_density_matmul_kernel",
    "density_matmul_route_stats",
    "reset_density_matmul_route_stats",
)
