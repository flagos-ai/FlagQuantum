"""Catalog authorization for the fused local CNOT sequence kernel."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from .kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-003-B"


def _cx_sequence_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the fused CNOT sequence against its exact catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.cnot_sequence.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_cx_sequence_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _cx_sequence_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused CNOT sequence kernel",
    )


def _apply_cataloged_cx_sequence(
    state: torch.Tensor,
    *,
    control_masks: torch.Tensor,
    target_masks: torch.Tensor,
    reverse_control_masks: torch.Tensor,
    reverse_target_masks: torch.Tensor,
    n_wires: int,
) -> torch.Tensor:
    """Execute the fused sequence after exact catalog authorization."""

    _require_cx_sequence_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton import cx_sequence

    return cx_sequence(
        state,
        control_masks=control_masks,
        target_masks=target_masks,
        reverse_control_masks=reverse_control_masks,
        reverse_target_masks=reverse_target_masks,
        n_wires=n_wires,
    )


__all__ = (
    "_apply_cataloged_cx_sequence",
    "_cx_sequence_kernel_match",
    "_require_cx_sequence_kernel",
)
