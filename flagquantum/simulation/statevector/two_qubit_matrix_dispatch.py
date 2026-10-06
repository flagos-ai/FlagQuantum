"""Catalog authorization and rollout policy for dense local two-qubit gates."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import torch

from ...kernels.catalog import (
    KernelImplementation,
    KernelMatchResult,
    KernelRequest,
    match_kernel_implementations,
)
from ..kernel_dispatch import _require_cataloged_kernel

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-009-A"
_MIN_AMPLITUDES_PER_STATE = 1 << 16
_MAX_AMPLITUDES_PER_STATE = 1 << 24
_EVIDENCED_BATCHES = frozenset((1, 4))


def _two_qubit_matrix_dispatch_enabled() -> bool:
    """Return whether the measured SV-009 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_TWO_QUBIT_MATRIX",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _two_qubit_matrix_shape_supported(
    state_shape: Sequence[int],
    matrix_shape: Sequence[int],
    *,
    wires: Sequence[int],
    n_wires: int,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    matrix_shape = tuple(int(item) for item in matrix_shape)
    wires = tuple(int(wire) for wire in wires)
    n_wires = int(n_wires)
    if (
        len(state_shape) != 2
        or matrix_shape != (4, 4)
        or len(wires) != 2
        or n_wires < 2
    ):
        return False
    batch, amplitudes = state_shape
    return bool(
        batch in _EVIDENCED_BATCHES
        and _MIN_AMPLITUDES_PER_STATE <= amplitudes <= _MAX_AMPLITUDES_PER_STATE
        and amplitudes == 1 << n_wires
        and wires[0] != wires[1]
        and all(0 <= wire < n_wires for wire in wires)
    )


def _two_qubit_matrix_kernel_enabled(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    wires: Sequence[int],
    n_wires: int,
) -> bool:
    """Return whether SV-009 supports this exact public runtime request."""

    if not _two_qubit_matrix_dispatch_enabled():
        return False
    if not _two_qubit_matrix_shape_supported(
        state.shape,
        matrix.shape,
        wires=wires,
        n_wires=n_wires,
    ):
        return False
    return bool(
        state.is_cuda
        and matrix.is_cuda
        and state.device == matrix.device
        and state.dtype == torch.complex64
        and matrix.dtype == state.dtype
        and state.is_contiguous()
        and matrix.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
        and not matrix.is_conj()
        and not matrix.is_neg()
        and not state.requires_grad
        and not matrix.requires_grad
    )


def _two_qubit_matrix_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the local dense two-qubit kernel against its exact catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.matrix_2q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_two_qubit_matrix_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-009 implementation or fail closed."""

    return _require_cataloged_kernel(
        _two_qubit_matrix_kernel_match(
            device_type=device_type,
            dtype=dtype,
        ),
        implementation_id=_IMPLEMENTATION_ID,
        description="dense local two-qubit matrix kernel",
    )


def _apply_cataloged_two_qubit_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Execute SV-009 after exact catalog authorization."""

    normalized_wires = tuple(int(wire) for wire in wires)
    if not _two_qubit_matrix_kernel_enabled(
        state,
        matrix,
        wires=normalized_wires,
        n_wires=n_wires,
    ):
        raise ValueError("dense local two-qubit kernel does not support this request")
    _require_two_qubit_matrix_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_gates import apply_complex64_local_2q

    return apply_complex64_local_2q(
        state,
        matrix,
        first_bit_position=int(n_wires) - 1 - normalized_wires[0],
        second_bit_position=int(n_wires) - 1 - normalized_wires[1],
    )


def _try_apply_cataloged_two_qubit_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor | None:
    """Route an evidenced dense request or preserve the reference path."""

    if not _two_qubit_matrix_kernel_enabled(
        state,
        matrix,
        wires=wires,
        n_wires=n_wires,
    ):
        return None
    return _apply_cataloged_two_qubit_matrix(
        state,
        matrix,
        wires=wires,
        n_wires=n_wires,
    )


__all__ = (
    "_apply_cataloged_two_qubit_matrix",
    "_require_two_qubit_matrix_kernel",
    "_try_apply_cataloged_two_qubit_matrix",
    "_two_qubit_matrix_dispatch_enabled",
    "_two_qubit_matrix_kernel_enabled",
    "_two_qubit_matrix_kernel_match",
    "_two_qubit_matrix_shape_supported",
)
