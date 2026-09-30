"""Catalog authorization for the fused local RX/RZ sequence kernel."""

from __future__ import annotations

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)

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
    """Return the wired implementation or fail closed on catalog drift."""

    match = _rx_rz_sequence_kernel_match(device_type=device_type, dtype=dtype)
    candidate = next(
        (
            item.implementation
            for item in match.candidates
            if item.implementation.implementation_id == _IMPLEMENTATION_ID
        ),
        None,
    )
    if candidate is not None:
        return candidate

    rejection = next(
        (
            item
            for item in match.rejections
            if item.implementation.implementation_id == _IMPLEMENTATION_ID
        ),
        None,
    )
    mismatch_codes = (
        tuple(mismatch.code for mismatch in rejection.mismatches)
        if rejection is not None
        else ("implementation_not_registered",)
    )
    raise RuntimeError(
        "fused RX/RZ sequence kernel is not authorized by the kernel catalog: "
        + ", ".join(mismatch_codes)
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
