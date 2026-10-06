"""Catalog authorization for fused sampled-wire MPS collapse updates."""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel
from .site_kernels import (
    _record_mps_sampling_collapse_fallback,
    _record_mps_sampling_collapse_route,
)

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-008-A"
_MAX_BOND = 64


def _mps_sampling_collapse_dispatch_enabled() -> bool:
    """Return whether the evidenced MPS-008 rollout is enabled."""

    return os.getenv("FQ_TRITON_MPS_SAMPLING_COLLAPSE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _mps_sampling_collapse_kernel_enabled(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> bool:
    """Return whether MPS-008 supports this exact sampling step."""

    if (
        not _mps_sampling_collapse_dispatch_enabled()
        or site.ndim != 4
        or next_site.ndim != 4
        or tuple(site.shape[1:3]) != (1, 2)
        or int(next_site.shape[2]) != 2
    ):
        return False
    batch = int(site.shape[0])
    right_dim = int(site.shape[-1])
    next_right_dim = int(next_site.shape[-1])
    return bool(
        site.is_cuda
        and site.dtype == torch.complex64
        and next_site.dtype == site.dtype
        and bits.device == site.device
        and next_site.device == site.device
        and bits.dtype in (torch.int32, torch.int64)
        and bits.shape == (batch,)
        and not site.requires_grad
        and not next_site.requires_grad
        and site.is_contiguous()
        and next_site.is_contiguous()
        and bits.is_contiguous()
        and not site.is_conj()
        and not site.is_neg()
        and not next_site.is_conj()
        and not next_site.is_neg()
        and batch > 0
        and right_dim > 0
        and next_right_dim > 0
        and int(next_site.shape[0]) == batch
        and int(next_site.shape[1]) == right_dim
        and right_dim <= _MAX_BOND
        and next_right_dim <= _MAX_BOND
    )


def _mps_sampling_collapse_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match a sampled-wire collapse against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.sampling.collapse_wire.local",
            device=device_type,
            dtype=dtype,
            layout="mps_sampling_step",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_mps_sampling_collapse_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected MPS-008 implementation or fail closed."""

    return _require_cataloged_kernel(
        _mps_sampling_collapse_kernel_match(
            device_type=device_type,
            dtype=dtype,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused MPS sampled-wire collapse kernel",
    )


def _apply_cataloged_mps_sampling_collapse(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute MPS-008 after exact catalog authorization."""

    _require_mps_sampling_collapse_kernel(
        device_type=site.device.type,
        dtype=str(site.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_sampling_collapse import (
        fused_mps_sampling_collapse,
    )

    return fused_mps_sampling_collapse(site, next_site, bits)


def _try_apply_cataloged_mps_sampling_collapse(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Route an evidenced sampling step or preserve the reference path."""

    if not _mps_sampling_collapse_kernel_enabled(site, next_site, bits):
        _record_mps_sampling_collapse_fallback()
        return None
    result = _apply_cataloged_mps_sampling_collapse(site, next_site, bits)
    _record_mps_sampling_collapse_route()
    return result


__all__ = (
    "_apply_cataloged_mps_sampling_collapse",
    "_mps_sampling_collapse_dispatch_enabled",
    "_mps_sampling_collapse_kernel_enabled",
    "_mps_sampling_collapse_kernel_match",
    "_require_mps_sampling_collapse_kernel",
    "_try_apply_cataloged_mps_sampling_collapse",
)
