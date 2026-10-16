"""Catalog authorization for the Hermitian MPS observable-adjoint VJP."""

from __future__ import annotations

import os

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-006-A"
_MAX_BOND = 64
_MAX_CONTRACTION_WORK = 1 << 25


def _mps_observable_adjoint_dispatch_enabled() -> bool:
    """Return whether the evidenced MPS-006 rollout is enabled."""

    return os.getenv("FQ_TRITON_MPS_OBSERVABLE_ADJOINT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _mps_observable_adjoint_kernel_enabled(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
    *,
    hermitian: bool,
) -> bool:
    """Return whether MPS-006 supports this exact local VJP request."""

    if (
        not _mps_observable_adjoint_dispatch_enabled()
        or not hermitian
        or tensor.ndim != 4
        or left_environment.ndim != 3
        or right_environment.ndim != 3
        or operator.ndim != 2
        or weights.ndim != 1
    ):
        return False
    batch, left_dim, physical_dim, right_dim = tensor.shape
    if (
        physical_dim != 2
        or left_environment.shape != (batch, left_dim, left_dim)
        or right_environment.shape != (batch, right_dim, right_dim)
        or operator.shape != (2, 2)
        or weights.shape != (batch,)
    ):
        return False
    contraction_elements = left_dim * 2 * right_dim
    contraction_work = batch * contraction_elements * contraction_elements
    complex_inputs = (tensor, left_environment, right_environment, operator)
    inputs = (*complex_inputs, weights)
    return bool(
        tensor.is_cuda
        and tensor.dtype == torch.complex64
        and all(item.device == tensor.device for item in inputs)
        and all(item.dtype == torch.complex64 for item in complex_inputs)
        and weights.dtype == torch.float32
        and all(item.is_contiguous() for item in inputs)
        and all(not item.is_conj() and not item.is_neg() for item in inputs)
        and batch > 0
        and 0 < left_dim <= _MAX_BOND
        and 0 < right_dim <= _MAX_BOND
        and contraction_work <= _MAX_CONTRACTION_WORK
    )


def _mps_observable_adjoint_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the local observable VJP against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.gradient.hermitian_observable_adjoint.local",
            device=device_type,
            dtype=dtype,
            layout="mps_local_observable",
            direction="vjp",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_mps_observable_adjoint_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected MPS-006 implementation or fail closed."""

    return _require_cataloged_kernel(
        _mps_observable_adjoint_kernel_match(
            device_type=device_type,
            dtype=dtype,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused Hermitian MPS observable-adjoint kernel",
    )


def _apply_cataloged_mps_observable_adjoint(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Execute MPS-006 after exact catalog authorization."""

    _require_mps_observable_adjoint_kernel(
        device_type=tensor.device.type,
        dtype=str(tensor.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_observable_adjoint import (
        fused_mps_hermitian_observable_adjoint,
    )

    return fused_mps_hermitian_observable_adjoint(
        tensor,
        left_environment,
        right_environment,
        operator,
        weights,
    )


def _try_apply_cataloged_mps_observable_adjoint(
    tensor: torch.Tensor,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    operator: torch.Tensor,
    weights: torch.Tensor,
    *,
    hermitian: bool,
) -> torch.Tensor | None:
    """Route an evidenced Hermitian local VJP or preserve the reference path."""

    if not _mps_observable_adjoint_kernel_enabled(
        tensor,
        left_environment,
        right_environment,
        operator,
        weights,
        hermitian=hermitian,
    ):
        return None
    return _apply_cataloged_mps_observable_adjoint(
        tensor,
        left_environment,
        right_environment,
        operator,
        weights,
    )


__all__ = (
    "_apply_cataloged_mps_observable_adjoint",
    "_mps_observable_adjoint_dispatch_enabled",
    "_mps_observable_adjoint_kernel_enabled",
    "_mps_observable_adjoint_kernel_match",
    "_require_mps_observable_adjoint_kernel",
    "_try_apply_cataloged_mps_observable_adjoint",
)
