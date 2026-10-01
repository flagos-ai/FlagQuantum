"""Catalog authorization for the local single-qubit matrix kernel."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-001-B"


def _single_qubit_matrix_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the differentiable matrix kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.matrix_1q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_single_qubit_matrix_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the differentiable implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _single_qubit_matrix_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="single-qubit matrix kernel",
    )


def _apply_cataloged_single_qubit_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Execute the differentiable matrix kernel after catalog authorization."""

    _require_single_qubit_matrix_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton import single_qubit_matrix

    return single_qubit_matrix(
        state,
        matrix,
        wire=wire,
        n_wires=n_wires,
    )


__all__ = (
    "_apply_cataloged_single_qubit_matrix",
    "_require_single_qubit_matrix_kernel",
    "_single_qubit_matrix_kernel_match",
)
