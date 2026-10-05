"""Catalog authorization for the fused local RX/RZ sequence kernel."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-005-A"


def _rx_rz_sequence_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the fused RX/RZ sequence against its exact catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.rx_rz_sequence.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_rx_rz_sequence_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _rx_rz_sequence_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused RX/RZ sequence kernel",
    )


def _apply_cataloged_rx_rz_sequence(
    state: torch.Tensor,
    rx_angles: torch.Tensor,
    rz_angles: torch.Tensor,
) -> object:
    """Execute the fused sequence after exact catalog authorization."""

    if state.is_cuda:
        _require_rx_rz_sequence_kernel(
            device_type=state.device.type,
            dtype=str(state.dtype).removeprefix("torch."),
        )
    from ...kernels.triton import repeated_rx_rz

    return repeated_rx_rz(state, rx_angles, rz_angles)


__all__ = (
    "_apply_cataloged_rx_rz_sequence",
    "_require_rx_rz_sequence_kernel",
    "_rx_rz_sequence_kernel_match",
)
