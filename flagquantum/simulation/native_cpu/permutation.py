"""Dispatch boundary for native CPU statevector permutations."""

from __future__ import annotations

import os
from typing import cast

import torch

from .adjoint import _load_extension


def native_cpu_cx_adjoint_gather_available() -> bool:
    """Return whether the dual-state CX gather is enabled and loadable."""

    return (
        os.getenv("FQ_NATIVE_CPU_CX_ADJOINT_GATHER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and _load_extension()
    )


def fused_cx_adjoint_gather(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather ket and adjoint with one shared permutation-table traversal."""

    if (
        not native_cpu_cx_adjoint_gather_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or index.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or index.dtype not in {torch.int32, torch.int64}
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or index.shape != (ket.shape[1],)
        or not all(item.is_contiguous() for item in (ket, adjoint, index))
    ):
        return None
    with torch.no_grad():
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            torch.ops.flagquantum_native.fused_cx_adjoint_gather(ket, adjoint, index),
        )
