"""Catalog-backed dispatch for cross-shard control-subspace transport."""

from __future__ import annotations

import torch

from ....kernels.catalog import KernelRequest
from .kernel_dispatch import KernelDecision, select_cataloged_triton_kernel


def _triton_control_subspace_pack_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the cataloged control-one packing implementation."""

    return select_cataloged_triton_kernel(
        "control_subspace_pack",
        request=KernelRequest(
            semantic_id="statevector.transport.control_subspace_pack",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("distributed", "control_subspace"),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-007-A",
        requested=True,
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_control_subspace_unpack_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the cataloged control-one unpacking implementation."""

    return select_cataloged_triton_kernel(
        "control_subspace_unpack",
        request=KernelRequest(
            semantic_id="statevector.transport.control_subspace_unpack",
            device=device_type,
            dtype=dtype,
            layout="packed_subspace",
            direction="forward",
            addressing=("distributed", "control_subspace"),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-008-A",
        requested=True,
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_control_subspace_tensor_decisions(
    amplitudes: torch.Tensor,
    output: torch.Tensor,
) -> tuple[KernelDecision, KernelDecision]:
    """Select pack and unpack independently from their tensor contracts."""

    device_type = amplitudes.device.type
    dtype = str(amplitudes.dtype).removeprefix("torch.")
    return (
        _triton_control_subspace_pack_decision(
            runtime_supported=amplitudes.ndim == 2 and amplitudes.is_contiguous(),
            device_type=device_type,
            dtype=dtype,
        ),
        _triton_control_subspace_unpack_decision(
            runtime_supported=(
                output.ndim == 2
                and output.is_contiguous()
                and output.device == amplitudes.device
                and output.dtype == amplitudes.dtype
                and output.shape[0] == amplitudes.shape[0]
            ),
            device_type=device_type,
            dtype=dtype,
        ),
    )


__all__ = (
    "_triton_control_subspace_pack_decision",
    "_triton_control_subspace_tensor_decisions",
    "_triton_control_subspace_unpack_decision",
)
