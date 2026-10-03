"""Catalog authorization for full statevector-probability dispatch."""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from ..kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernels.catalog.schema import KernelDirection
from ..simulation.kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-MEAS-001-A"
_DISPATCH_AMPLITUDES_PER_STATE = 1 << 24
_DISPATCH_BATCH = 1


def _statevector_probability_dispatch_enabled() -> bool:
    """Return whether the measured MEAS-001 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_STATEVECTOR_PROBABILITIES",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _statevector_probability_kernel_enabled(state: torch.Tensor) -> bool:
    """Return whether MEAS-001 supports this exact runtime request."""

    if tuple(state.shape) != (
        _DISPATCH_BATCH,
        _DISPATCH_AMPLITUDES_PER_STATE,
    ):
        return False
    if not _statevector_probability_dispatch_enabled():
        return False
    return bool(
        state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
    )


def _statevector_probability_kernel_match(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelMatchResult:
    """Match one full statevector-probability request against the catalog."""

    if direction not in {"forward", "backward"}:
        raise ValueError(
            "statevector probability direction must be forward or backward"
        )
    return match_kernel_implementations(
        KernelRequest(
            semantic_id="measurement.probabilities.statevector",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction=direction,
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_statevector_probability_kernel(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelImplementation:
    """Return the wired MEAS-001 implementation or fail closed."""

    return _require_cataloged_kernel(
        _statevector_probability_kernel_match(
            device_type=device_type,
            dtype=dtype,
            direction=direction,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="statevector probability kernel",
    )


def _apply_cataloged_statevector_probabilities(state: torch.Tensor) -> torch.Tensor:
    """Execute MEAS-001 after exact catalog authorization."""

    _require_statevector_probability_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
        direction="backward" if state.requires_grad else "forward",
    )
    from ..kernels.triton.statevector_measurement import statevector_probabilities

    return statevector_probabilities(state)


def _try_apply_cataloged_statevector_probabilities(
    state: torch.Tensor,
) -> torch.Tensor | None:
    """Route an evidenced full-distribution request or preserve the reference."""

    if not _statevector_probability_kernel_enabled(state):
        return None
    return _apply_cataloged_statevector_probabilities(state)


__all__ = (
    "_apply_cataloged_statevector_probabilities",
    "_require_statevector_probability_kernel",
    "_statevector_probability_dispatch_enabled",
    "_statevector_probability_kernel_enabled",
    "_statevector_probability_kernel_match",
    "_try_apply_cataloged_statevector_probabilities",
)
