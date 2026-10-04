"""Catalog authorization for the fused local RY/RZ pair kernel."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-004-A"


def _ry_rz_pair_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the fused RY/RZ pair against its exact catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.ry_rz_pair.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_ry_rz_pair_kernel(*, device_type: str, dtype: str) -> KernelImplementation:
    """Return the connected implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _ry_rz_pair_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused RY/RZ pair kernel",
    )


def _apply_cataloged_ry_rz_pair(
    state: torch.Tensor,
    ry_angles: torch.Tensor,
    rz_angles: torch.Tensor,
    *,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Execute the fused pair after exact catalog authorization."""

    _require_ry_rz_pair_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton import ry_rz_pair

    return ry_rz_pair(
        state,
        ry_angles,
        rz_angles,
        qubit=wire,
        n_qubits=n_wires,
    )


__all__ = (
    "_apply_cataloged_ry_rz_pair",
    "_require_ry_rz_pair_kernel",
    "_ry_rz_pair_kernel_match",
)
