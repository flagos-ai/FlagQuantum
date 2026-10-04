"""Catalog authorization for statevector Pauli-product expectation dispatch."""

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
from ...kernels.catalog.schema import KernelDirection
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-MEAS-002-A"
_DISPATCH_AMPLITUDES_PER_STATE = 1 << 24
_DISPATCH_BATCH = 1
_DISPATCH_OPERATORS = ((0, "X"), (12, "Y"), (23, "Z"))


def _statevector_pauli_expectation_dispatch_enabled() -> bool:
    """Return whether the measured MEAS-002 statevector route is enabled."""

    return os.getenv(
        "FQ_TRITON_STATEVECTOR_PAULI_EXPECTATION",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _statevector_pauli_expectation_kernel_enabled(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
) -> bool:
    """Return whether MEAS-002 supports this exact evidenced request."""

    if (
        tuple(state.shape)
        != (
            _DISPATCH_BATCH,
            _DISPATCH_AMPLITUDES_PER_STATE,
        )
        or operators != _DISPATCH_OPERATORS
        or not _statevector_pauli_expectation_dispatch_enabled()
    ):
        return False
    return bool(
        state.is_cuda
        and state.dtype == torch.complex64
        and state.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
    )


def _statevector_pauli_expectation_kernel_match(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelMatchResult:
    """Match one statevector Pauli expectation against the catalog contract."""

    if direction not in {"forward", "backward"}:
        raise ValueError(
            "statevector Pauli expectation direction must be forward or backward"
        )
    return match_kernel_implementations(
        KernelRequest(
            semantic_id="measurement.expectation.pauli_product.statevector",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction=direction,
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_statevector_pauli_expectation_kernel(
    *, device_type: str, dtype: str, direction: KernelDirection
) -> KernelImplementation:
    """Return the wired MEAS-002 implementation or fail closed."""

    return _require_cataloged_kernel(
        _statevector_pauli_expectation_kernel_match(
            device_type=device_type,
            dtype=dtype,
            direction=direction,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="statevector Pauli-product expectation kernel",
    )


def _apply_cataloged_statevector_pauli_expectation(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
) -> torch.Tensor:
    """Execute MEAS-002 after exact catalog authorization."""

    _require_statevector_pauli_expectation_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
        direction="backward" if state.requires_grad else "forward",
    )
    from ...kernels.triton.statevector_measurement import (
        statevector_pauli_expectation,
    )

    return statevector_pauli_expectation(state, operators)


def _try_apply_cataloged_statevector_pauli_expectation(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
) -> torch.Tensor | None:
    """Route an evidenced Pauli request or preserve the reference path."""

    if not _statevector_pauli_expectation_kernel_enabled(state, operators):
        return None
    return _apply_cataloged_statevector_pauli_expectation(state, operators)


__all__ = (
    "_apply_cataloged_statevector_pauli_expectation",
    "_require_statevector_pauli_expectation_kernel",
    "_statevector_pauli_expectation_dispatch_enabled",
    "_statevector_pauli_expectation_kernel_enabled",
    "_statevector_pauli_expectation_kernel_match",
    "_try_apply_cataloged_statevector_pauli_expectation",
)
