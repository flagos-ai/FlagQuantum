"""Catalog authorization for fused MPS environment transfers."""

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

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-004-A"
_MAX_BOND = 32
_MAX_CONTRACTION_WORK = 1 << 22


def _mps_environment_kernel_enabled(
    environment: torch.Tensor,
    tensor: torch.Tensor,
) -> bool:
    """Return whether the opt-in route supports this exact tensor pair."""

    enabled = os.getenv("FQ_TRITON_MPS_ENVIRONMENT", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    if not enabled or environment.ndim != 3 or tensor.ndim != 4:
        return False
    batch, left_dim, physical_dim, right_dim = tensor.shape
    if physical_dim != 2 or environment.shape != (batch, left_dim, left_dim):
        return False
    contraction_work = batch * left_dim * left_dim * right_dim * right_dim
    return bool(
        environment.is_cuda
        and environment.dtype == torch.complex64
        and tensor.dtype == torch.complex64
        and tensor.device == environment.device
        and not environment.requires_grad
        and not tensor.requires_grad
        and environment.is_contiguous()
        and tensor.is_contiguous()
        and not environment.is_conj()
        and not environment.is_neg()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and batch > 0
        and 0 < left_dim <= _MAX_BOND
        and 0 < right_dim <= _MAX_BOND
        and contraction_work <= _MAX_CONTRACTION_WORK
    )


def _mps_environment_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the environment transfer against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.environment.transfer_identity_z",
            device=device_type,
            dtype=dtype,
            layout="mps_environment",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


def _require_mps_environment_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired implementation or fail closed on catalog drift."""

    return _require_cataloged_kernel(
        _mps_environment_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused MPS environment-transfer kernel",
    )


def _apply_cataloged_mps_environment(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool,
) -> torch.Tensor:
    """Execute the fused transfer after exact catalog authorization."""

    _require_mps_environment_kernel(
        device_type=environment.device.type,
        dtype=str(environment.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_environment import fused_mps_environment_transfer

    return fused_mps_environment_transfer(
        environment,
        tensor,
        insert_z=insert_z,
    )


def _try_apply_cataloged_mps_environment(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    *,
    insert_z: bool,
) -> torch.Tensor | None:
    """Route an evidenced input to MPS-004 or leave it to the reference path."""

    if not _mps_environment_kernel_enabled(environment, tensor):
        return None
    return _apply_cataloged_mps_environment(
        environment,
        tensor,
        insert_z=insert_z,
    )


__all__ = (
    "_apply_cataloged_mps_environment",
    "_mps_environment_kernel_enabled",
    "_mps_environment_kernel_match",
    "_require_mps_environment_kernel",
    "_try_apply_cataloged_mps_environment",
)
