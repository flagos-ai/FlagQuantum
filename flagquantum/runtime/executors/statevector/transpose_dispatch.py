"""Catalog-backed dispatch policy for fused distributed transpose gates."""

from __future__ import annotations

import os

import torch

from ....kernels.catalog import KernelRequest
from .kernel_dispatch import KernelDecision, select_cataloged_triton_kernel


def _triton_transpose_1q_requested() -> bool:
    return os.getenv("FQ_STATEVECTOR_TRITON_TRANSPOSE_1Q", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _triton_transpose_1q_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the exact evidenced transpose implementation connected at runtime."""

    return select_cataloged_triton_kernel(
        "transpose_1q",
        request=KernelRequest(
            semantic_id="statevector.distributed.transpose_apply_1q",
            device=device_type,
            dtype=dtype,
            layout="sharded_statevector",
            direction="forward",
            addressing=("distributed", "transpose"),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-006-A",
        requested=_triton_transpose_1q_requested(),
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_transpose_1q_tensor_decision(
    amplitudes: torch.Tensor,
    *,
    runtime_supported: bool = True,
) -> KernelDecision:
    return _triton_transpose_1q_decision(
        runtime_supported=runtime_supported and amplitudes.is_contiguous(),
        device_type=amplitudes.device.type,
        dtype=str(amplitudes.dtype).removeprefix("torch."),
    )


__all__ = (
    "_triton_transpose_1q_decision",
    "_triton_transpose_1q_tensor_decision",
)
