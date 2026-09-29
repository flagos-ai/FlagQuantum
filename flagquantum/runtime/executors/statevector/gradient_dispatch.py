"""Catalog-backed dispatch policy for statevector gradient kernels."""

from __future__ import annotations

import os

import torch

from ....kernels.catalog import KernelRequest
from .kernel_dispatch import (
    KernelDecision,
    select_cataloged_triton_kernel,
    select_triton_kernel,
)


def _triton_vjp_adjoint_requested() -> bool:
    return os.getenv("FQ_STATEVECTOR_TRITON_VJP_ADJOINT", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _triton_vjp_adjoint_decision(*, supported: bool = True) -> KernelDecision:
    """Preserve the generic route for reversible and sharded VJP kernels."""

    return select_triton_kernel(
        "vjp_adjoint",
        requested=_triton_vjp_adjoint_requested(),
        supported=supported,
    )


def _triton_local_adjoint_vjp_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the exact evidenced local adjoint VJP implementation."""

    return select_cataloged_triton_kernel(
        "local_adjoint_vjp",
        request=KernelRequest(
            semantic_id="gradient.vjp.adjoint_1q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="vjp",
            addressing=("local",),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-GR-001-A",
        requested=_triton_vjp_adjoint_requested(),
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_local_adjoint_vjp_tensor_decision(
    before: torch.Tensor,
    adjoint: torch.Tensor,
    *,
    runtime_supported: bool = True,
) -> KernelDecision:
    """Select the local adjoint VJP from its state tensor contract."""

    tensors_supported = bool(
        before.ndim == 2
        and before.is_contiguous()
        and adjoint.shape == before.shape
        and adjoint.device == before.device
        and adjoint.dtype == before.dtype
        and adjoint.is_contiguous()
    )
    return _triton_local_adjoint_vjp_decision(
        runtime_supported=runtime_supported and tensors_supported,
        device_type=before.device.type,
        dtype=str(before.dtype).removeprefix("torch."),
    )


def _triton_local_reversible_vjp_decision(
    *,
    runtime_supported: bool = True,
    device_type: str,
    dtype: str,
) -> KernelDecision:
    """Select the exact evidenced local reversible VJP implementation."""

    return select_cataloged_triton_kernel(
        "local_reversible_vjp",
        request=KernelRequest(
            semantic_id="gradient.vjp.reversible_1q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="vjp",
            addressing=("local",),
            providers=("triton",),
        ),
        implementation_id="FQKI-TRITON-GR-002-A",
        requested=_triton_vjp_adjoint_requested(),
        runtime_supported=runtime_supported,
        device_runtime_provider="pytorch",
        compiler_backend="cuda",
        capture_compiler_identity=True,
    )


def _triton_local_reversible_vjp_tensor_decision(
    ket: torch.Tensor | None,
    adjoint: torch.Tensor,
    *,
    runtime_supported: bool = True,
) -> KernelDecision:
    """Select the local reversible VJP from its state tensor contract."""

    tensors_supported = bool(
        ket is not None
        and ket.ndim == 2
        and ket.is_contiguous()
        and adjoint.shape == ket.shape
        and adjoint.device == ket.device
        and adjoint.dtype == ket.dtype
        and adjoint.is_contiguous()
        and adjoint.data_ptr() != ket.data_ptr()
    )
    state = adjoint if ket is None else ket
    return _triton_local_reversible_vjp_decision(
        runtime_supported=runtime_supported and tensors_supported,
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )


def _triton_adjoint_vjp_tensor_decision(
    before: torch.Tensor,
    adjoint: torch.Tensor,
    *,
    wire_count: int,
    sharded: bool,
    runtime_supported: bool = True,
) -> KernelDecision:
    """Use exact catalog identity only for the wired local one-qubit kernel."""

    if wire_count == 1 and not sharded:
        return _triton_local_adjoint_vjp_tensor_decision(
            before,
            adjoint,
            runtime_supported=runtime_supported,
        )
    return _triton_vjp_adjoint_decision(
        supported=bool(
            runtime_supported
            and wire_count == 1
            and before.dtype == torch.complex64
            and before.device.type == "cuda"
        )
    )


__all__ = (
    "_triton_adjoint_vjp_tensor_decision",
    "_triton_local_adjoint_vjp_decision",
    "_triton_local_adjoint_vjp_tensor_decision",
    "_triton_local_reversible_vjp_decision",
    "_triton_local_reversible_vjp_tensor_decision",
    "_triton_vjp_adjoint_decision",
)
