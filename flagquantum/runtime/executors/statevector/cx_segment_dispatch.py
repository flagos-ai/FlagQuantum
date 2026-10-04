"""Catalog-backed dispatch policy for fused local CNOT sequences."""

from __future__ import annotations

import os

import torch

from ....core.ir import CircuitIR
from ....kernels.catalog import KernelRequest
from .kernel_dispatch import KernelDecision, select_cataloged_triton_kernel


def _triton_local_cx_segment_requested(ir: CircuitIR | None = None) -> bool:
    raw = os.getenv("FQ_STATEVECTOR_TRITON_CX_SEGMENT", "")
    if not raw:
        return bool(
            ir is not None
            and ir.metadata.get("statevector_dependency_schedule_changed", False)
        )
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _triton_local_cx_segment_decision(
    ir: CircuitIR | None = None,
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the exact evidenced implementation connected into the runtime."""

    return select_cataloged_triton_kernel(
        "local_cx_segment",
        request=KernelRequest(
            semantic_id="statevector.apply.cnot_sequence.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-SV-003-A",
        requested=_triton_local_cx_segment_requested(ir),
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_local_cx_segment_tensor_decision(
    ir: CircuitIR | None,
    amplitudes: torch.Tensor,
    *,
    runtime_supported: bool = True,
) -> KernelDecision:
    return _triton_local_cx_segment_decision(
        ir,
        runtime_supported=runtime_supported and amplitudes.is_contiguous(),
        device_type=amplitudes.device.type,
        dtype=str(amplitudes.dtype).removeprefix("torch."),
    )


def _triton_local_cx_segment_enabled(
    ir: CircuitIR | None = None,
    *,
    device_type: str = "cuda",
    dtype: str = "complex64",
) -> bool:
    return _triton_local_cx_segment_decision(
        ir,
        device_type=device_type,
        dtype=dtype,
    ).accelerated


__all__ = (
    "_triton_local_cx_segment_decision",
    "_triton_local_cx_segment_enabled",
    "_triton_local_cx_segment_tensor_decision",
)
