"""Catalog authorization and rollout policy for local diagonal gates."""

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
from ..numerics.complex_arithmetic import complex_mul

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-010-A"
_MIN_AMPLITUDES_PER_STATE = 1 << 16
_MAX_AMPLITUDES_PER_STATE = 1 << 24
_EVIDENCED_BATCHES = frozenset((1, 4))


@lru_cache(maxsize=None)
def _diagonal_layout(
    n_wires: int, wires: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the statevector permutation used by the reference route."""

    rest_wires = tuple(wire for wire in range(n_wires) if wire not in wires)
    permutation = (
        (0,)
        + tuple(wire + 1 for wire in wires)
        + tuple(wire + 1 for wire in rest_wires)
    )
    inverse = [0] * len(permutation)
    for index, axis in enumerate(permutation):
        inverse[axis] = index
    return permutation, tuple(inverse)


def _apply_diagonal_matrix_reference(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Apply a diagonal matrix through the established PyTorch reference path."""

    wires = tuple(wires)
    batch = state.shape[0]
    dimension = 2 ** len(wires)
    if state.device.type == "cpu" and state.is_contiguous():
        matrix = matrix.to(device=state.device, dtype=state.dtype)
        diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
        if diagonal.ndim == 1:
            diagonal = diagonal.unsqueeze(0).expand(batch, -1)
        elif diagonal.shape[0] == 1 and batch != 1:
            diagonal = diagonal.expand(batch, -1)
        if diagonal.shape != (batch, dimension):
            raise ValueError(
                "diagonal matrix must have shape [dim, dim] or [batch, dim, dim]"
            )
        ordered_wires = tuple(sorted(wires))
        wire_order = tuple(wires.index(wire) for wire in ordered_wires)
        factors = diagonal.reshape((batch,) + (2,) * len(wires)).permute(
            (0,) + tuple(index + 1 for index in wire_order)
        )
        factor_shape = [batch] + [1] * n_wires
        for wire in ordered_wires:
            factor_shape[wire + 1] = 2
        tensor = state.reshape((batch,) + (2,) * n_wires)
        return (tensor * factors.reshape(factor_shape)).reshape(state.shape)
    if layout is None:
        layout = _diagonal_layout(n_wires, wires)
    permutation, inverse_permutation = layout
    tensor = state.reshape((batch,) + (2,) * n_wires).permute(permutation)
    flat = tensor.reshape(batch, dimension, -1)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    if diagonal.ndim == 1:
        diagonal = diagonal.expand(batch, -1)
    output = complex_mul(flat, diagonal.unsqueeze(-1))
    return (
        output.reshape((batch,) + (2,) * n_wires)
        .permute(inverse_permutation)
        .reshape(batch, -1)
    )


def _diagonal_matrix_dispatch_enabled() -> bool:
    """Return whether the measured SV-010 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_DIAGONAL_MATRIX",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _diagonal_matrix_shape_supported(
    state_shape: Sequence[int],
    matrix_shape: Sequence[int],
    *,
    qubits: Sequence[int],
    n_qubits: int,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    matrix_shape = tuple(int(item) for item in matrix_shape)
    qubits = tuple(int(qubit) for qubit in qubits)
    n_qubits = int(n_qubits)
    if len(state_shape) != 2 or len(qubits) not in {1, 2} or n_qubits < len(qubits):
        return False
    batch, amplitudes = state_shape
    operator_size = 1 << len(qubits)
    matrix_supported = matrix_shape == (operator_size, operator_size) or (
        len(matrix_shape) == 3
        and matrix_shape[0] in {1, batch}
        and matrix_shape[1:] == (operator_size, operator_size)
    )
    return bool(
        matrix_supported
        and batch in _EVIDENCED_BATCHES
        and _MIN_AMPLITUDES_PER_STATE <= amplitudes <= _MAX_AMPLITUDES_PER_STATE
        and amplitudes == 1 << n_qubits
        and len(set(qubits)) == len(qubits)
        and all(0 <= qubit < n_qubits for qubit in qubits)
    )


def _diagonal_matrix_kernel_enabled(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    qubits: Sequence[int],
    n_qubits: int,
) -> bool:
    """Return whether SV-010 supports this exact public runtime request."""

    return bool(
        _diagonal_matrix_dispatch_enabled()
        and _diagonal_matrix_shape_supported(
            state.shape,
            matrix.shape,
            qubits=qubits,
            n_qubits=n_qubits,
        )
        and state.is_cuda
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


def _diagonal_matrix_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the local diagonal kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.diagonal.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_diagonal_matrix_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-010 implementation or fail closed."""

    return _require_cataloged_kernel(
        _diagonal_matrix_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="local diagonal matrix kernel",
    )


def _apply_cataloged_diagonal_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    qubits: Sequence[int],
) -> torch.Tensor:
    """Execute SV-010 after exact catalog authorization."""

    normalized_qubits = tuple(int(qubit) for qubit in qubits)
    _require_diagonal_matrix_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_diagonal import apply_complex64_local_diagonal

    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    return apply_complex64_local_diagonal(
        state,
        diagonal,
        qubits=normalized_qubits,
    )


def _try_apply_cataloged_diagonal_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor | None:
    """Route an evidenced diagonal request or preserve the reference path."""

    if not _diagonal_matrix_kernel_enabled(
        state,
        matrix,
        qubits=qubits,
        n_qubits=n_qubits,
    ):
        return None
    return _apply_cataloged_diagonal_matrix(state, matrix, qubits=qubits)


def _apply_diagonal_matrix(
    state: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Apply a diagonal matrix through catalog dispatch or the reference path."""

    wires = tuple(wires)
    dispatched = _try_apply_cataloged_diagonal_matrix(
        state,
        matrix,
        qubits=wires,
        n_qubits=n_wires,
    )
    if dispatched is not None:
        return dispatched
    return _apply_diagonal_matrix_reference(
        state,
        matrix,
        wires,
        n_wires,
        layout=layout,
    )


__all__ = (
    "_apply_diagonal_matrix",
    "_apply_diagonal_matrix_reference",
    "_apply_cataloged_diagonal_matrix",
    "_diagonal_matrix_dispatch_enabled",
    "_diagonal_matrix_kernel_enabled",
    "_diagonal_matrix_kernel_match",
    "_diagonal_matrix_shape_supported",
    "_require_diagonal_matrix_kernel",
    "_try_apply_cataloged_diagonal_matrix",
)
