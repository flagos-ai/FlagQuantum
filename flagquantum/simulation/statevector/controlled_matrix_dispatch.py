"""Catalog authorization and rollout policy for controlled one-qubit rotations."""

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
from ..matrices import rx_mat, ry_mat, rz_mat

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-012-A"
_EVIDENCED_STATE_SHAPES = frozenset(
    {
        (1, 1 << 20),
        (1, 1 << 24),
        (4, 1 << 20),
    }
)
_TARGET_MATRIX_BY_OPCODE = {"crx": rx_mat, "cry": ry_mat, "crz": rz_mat}


def _controlled_matrix_dispatch_enabled() -> bool:
    """Return whether the measured SV-012 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_CONTROLLED_MATRIX_1Q",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _controlled_matrix_shape_supported(
    state_shape: Sequence[int],
    angle_shape: Sequence[int],
    *,
    control_qubit: int,
    target_qubit: int,
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    angle_shape = tuple(int(item) for item in angle_shape)
    control_qubit = int(control_qubit)
    target_qubit = int(target_qubit)
    n_qubits = int(n_qubits)
    if len(state_shape) != 2 or n_qubits < 2:
        return False
    batch, amplitudes = state_shape
    return bool(
        opcode in _TARGET_MATRIX_BY_OPCODE
        and state_shape in _EVIDENCED_STATE_SHAPES
        and amplitudes == 1 << n_qubits
        and angle_shape in {(1,), (batch,)}
        and control_qubit != target_qubit
        and 0 <= control_qubit < n_qubits
        and 0 <= target_qubit < n_qubits
    )


def _controlled_matrix_kernel_enabled(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    control_qubit: int,
    target_qubit: int,
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether SV-012 supports this exact public runtime request."""

    return bool(
        _controlled_matrix_dispatch_enabled()
        and _controlled_matrix_shape_supported(
            state.shape,
            angles.shape,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            n_qubits=n_qubits,
            opcode=opcode,
        )
        and state.is_cuda
        and angles.is_cuda
        and state.device == angles.device
        and state.dtype == torch.complex64
        and angles.dtype == torch.float32
        and state.is_contiguous()
        and angles.is_contiguous()
        and not state.is_conj()
        and not state.is_neg()
        and not angles.is_neg()
        and not state.requires_grad
        and not angles.requires_grad
    )


def _controlled_matrix_kernel_match(
    *, device_type: str, dtype: str
) -> KernelMatchResult:
    """Match the controlled-matrix kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.controlled_matrix_1q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_controlled_matrix_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-012 implementation or fail closed."""

    return _require_cataloged_kernel(
        _controlled_matrix_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="controlled one-qubit matrix kernel",
    )


def _apply_cataloged_controlled_rotation(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    control_qubit: int,
    target_qubit: int,
    opcode: str,
) -> torch.Tensor:
    """Materialize only the target matrix and execute cataloged SV-012."""

    _require_controlled_matrix_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_controlled_matrix import (
        apply_complex64_local_controlled_1q,
    )

    matrix = _TARGET_MATRIX_BY_OPCODE[opcode](angles).contiguous()
    if angles.shape == (1,):
        matrix = matrix[0]
    return apply_complex64_local_controlled_1q(
        state,
        matrix,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
    )


def _try_apply_cataloged_controlled_rotation(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    control_qubit: int,
    target_qubit: int,
    n_qubits: int,
    opcode: str,
) -> torch.Tensor | None:
    """Route an evidenced controlled rotation or preserve the reference path."""

    if not _controlled_matrix_kernel_enabled(
        state,
        angles,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        n_qubits=n_qubits,
        opcode=opcode,
    ):
        return None
    return _apply_cataloged_controlled_rotation(
        state,
        angles,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        opcode=opcode,
    )


__all__ = (
    "_apply_cataloged_controlled_rotation",
    "_controlled_matrix_dispatch_enabled",
    "_controlled_matrix_kernel_enabled",
    "_controlled_matrix_kernel_match",
    "_controlled_matrix_shape_supported",
    "_require_controlled_matrix_kernel",
    "_try_apply_cataloged_controlled_rotation",
)
