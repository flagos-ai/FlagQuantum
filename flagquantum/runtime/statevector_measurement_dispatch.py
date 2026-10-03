"""Catalog authorization for statevector marginal-probability dispatch."""

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

_IMPLEMENTATION_ID = "FQKI-TRITON-MEAS-003-A"
_MAX_QUBITS = 30
_DISPATCH_AMPLITUDES_PER_STATE = 1 << 24
_DISPATCH_BATCH = 1
_DISPATCH_SELECTED_QUBITS = 4


def _statevector_marginal_dispatch_enabled() -> bool:
    """Return whether the measured MEAS-003 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_STATEVECTOR_MARGINAL_PROBABILITIES",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _statevector_marginal_kernel_enabled(
    state: torch.Tensor,
    qubits: tuple[int, ...],
) -> bool:
    """Return whether MEAS-003 supports this exact runtime request."""

    if not _statevector_marginal_dispatch_enabled() or state.ndim != 2:
        return False
    amplitudes = int(state.shape[1])
    n_qubits = amplitudes.bit_length() - 1
    return bool(
        state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
        and int(state.shape[0]) == _DISPATCH_BATCH
        and 2**n_qubits == amplitudes
        and n_qubits <= _MAX_QUBITS
        and len(qubits) == _DISPATCH_SELECTED_QUBITS
        and amplitudes == _DISPATCH_AMPLITUDES_PER_STATE
    )


def _statevector_marginal_kernel_match(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelMatchResult:
    """Match one statevector marginal request against the catalog contract."""

    if direction not in {"forward", "backward"}:
        raise ValueError("statevector marginal direction must be forward or backward")
    return match_kernel_implementations(
        KernelRequest(
            semantic_id="measurement.probabilities.marginal.statevector",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction=direction,
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_statevector_marginal_kernel(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelImplementation:
    """Return the wired MEAS-003 implementation or fail closed."""

    return _require_cataloged_kernel(
        _statevector_marginal_kernel_match(
            device_type=device_type,
            dtype=dtype,
            direction=direction,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="statevector marginal-probability kernel",
    )


def _apply_cataloged_statevector_marginal(
    state: torch.Tensor,
    qubits: tuple[int, ...],
) -> torch.Tensor:
    """Execute MEAS-003 after exact catalog authorization."""

    _require_statevector_marginal_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
        direction="backward" if state.requires_grad else "forward",
    )
    from ..kernels.triton.statevector_measurement import (
        statevector_marginal_probabilities,
    )

    return statevector_marginal_probabilities(state, qubits)


def _try_apply_cataloged_statevector_marginal(
    state: torch.Tensor,
    qubits: tuple[int, ...],
) -> torch.Tensor | None:
    """Route an evidenced statevector request or preserve the reference path."""

    if not _statevector_marginal_kernel_enabled(state, qubits):
        return None
    return _apply_cataloged_statevector_marginal(state, qubits)


__all__ = (
    "_apply_cataloged_statevector_marginal",
    "_require_statevector_marginal_kernel",
    "_statevector_marginal_dispatch_enabled",
    "_statevector_marginal_kernel_enabled",
    "_statevector_marginal_kernel_match",
    "_try_apply_cataloged_statevector_marginal",
)
