"""Catalog authorization for fused MPS qubit-probability reductions."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel
from .site_kernels import (
    _record_mps_qubit_probability_fallback,
    _record_mps_qubit_probability_route,
)

if TYPE_CHECKING:
    from .state import MPSState

_IMPLEMENTATION_ID = "FQKI-TRITON-MPS-007-A"
_MAX_SITE_ELEMENTS = 1 << 12


def _mps_wire_probability_dispatch_enabled() -> bool:
    return os.getenv("FQ_TRITON_MPS_WIRE_PROBABILITIES", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _mps_wire_probability_kernel_enabled(tensor: torch.Tensor) -> bool:
    """Return whether MPS-007 supports this exact site tensor."""

    if (
        not _mps_wire_probability_dispatch_enabled()
        or tensor.ndim != 4
        or int(tensor.shape[2]) != 2
    ):
        return False
    batch, left_dim, _, right_dim = tensor.shape
    return bool(
        tensor.is_cuda
        and tensor.dtype == torch.complex64
        and not tensor.requires_grad
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and batch > 0
        and left_dim > 0
        and right_dim > 0
        and left_dim * right_dim <= _MAX_SITE_ELEMENTS
    )


def _mps_wire_probability_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the local probability reduction against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="mps.measurement.wire_probabilities.local",
            device=device_type,
            dtype=dtype,
            layout="mps_site_tensor",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_mps_wire_probability_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the wired MPS-007 implementation or fail closed."""

    return _require_cataloged_kernel(
        _mps_wire_probability_kernel_match(
            device_type=device_type,
            dtype=dtype,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="fused MPS qubit-probability kernel",
    )


def _apply_cataloged_mps_wire_probabilities(tensor: torch.Tensor) -> torch.Tensor:
    """Execute MPS-007 after exact catalog authorization."""

    _require_mps_wire_probability_kernel(
        device_type=tensor.device.type,
        dtype=str(tensor.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.mps_wire_probabilities import (
        fused_mps_wire_probabilities,
    )

    return fused_mps_wire_probabilities(tensor)


def _try_apply_cataloged_mps_wire_probabilities(
    tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Route an evidenced site tensor or preserve the reference path."""

    if not _mps_wire_probability_kernel_enabled(tensor):
        return None
    return _apply_cataloged_mps_wire_probabilities(tensor)


def _mps_wire_probabilities(state: MPSState, qubit: int) -> torch.Tensor:
    """Evaluate one sampling probability pair through MPS-007 or PyTorch."""

    state.move_orthogonality_center(qubit)
    tensor = state.tensors[qubit]
    probabilities = _try_apply_cataloged_mps_wire_probabilities(tensor)
    if probabilities is None:
        _record_mps_qubit_probability_fallback()
        probabilities = torch.sum(torch.abs(tensor) ** 2, dim=(1, 3))
        probabilities = torch.clamp(probabilities, min=0)
    else:
        _record_mps_qubit_probability_route()
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    if bool(torch.any(~torch.isfinite(probabilities))) or bool(
        torch.any(normalizer <= 1e-12)
    ):
        raise RuntimeError("MPS measurement probabilities are not finite")
    return probabilities / torch.clamp(normalizer, min=1e-12)


__all__ = (
    "_apply_cataloged_mps_wire_probabilities",
    "_mps_wire_probability_dispatch_enabled",
    "_mps_wire_probability_kernel_enabled",
    "_mps_wire_probability_kernel_match",
    "_mps_wire_probabilities",
    "_require_mps_wire_probability_kernel",
    "_try_apply_cataloged_mps_wire_probabilities",
)
