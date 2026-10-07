"""Catalog authorization and rollout policy for two-qubit Pauli rotations."""

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

_IMPLEMENTATION_ID = "FQKI-TRITON-SV-011-A"
_EVIDENCED_STATE_SHAPES = frozenset(
    {
        (1, 1 << 20),
        (1, 1 << 24),
        (4, 1 << 20),
    }
)
_PAULI_BY_OPCODE = {"rxx": "XX", "ryy": "YY", "rzz": "ZZ"}


def _pauli_rotation_dispatch_enabled() -> bool:
    """Return whether the measured SV-011 runtime route is enabled."""

    return os.getenv(
        "FQ_TRITON_PAULI_ROTATION_2Q",
        "1",
    ).strip().lower() not in {"0", "false", "off", "no"}


def _pauli_rotation_shape_supported(
    state_shape: Sequence[int],
    angle_shape: Sequence[int],
    *,
    qubits: Sequence[int],
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether structural inputs stay inside the measured rollout window."""

    state_shape = tuple(int(item) for item in state_shape)
    angle_shape = tuple(int(item) for item in angle_shape)
    qubits = tuple(int(qubit) for qubit in qubits)
    n_qubits = int(n_qubits)
    if len(state_shape) != 2 or len(qubits) != 2 or n_qubits < 2:
        return False
    batch, amplitudes = state_shape
    return bool(
        opcode in _PAULI_BY_OPCODE
        and state_shape in _EVIDENCED_STATE_SHAPES
        and amplitudes == 1 << n_qubits
        and angle_shape in {(1,), (batch,)}
        and len(set(qubits)) == 2
        and all(0 <= qubit < n_qubits for qubit in qubits)
    )


def _pauli_rotation_kernel_enabled(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    qubits: Sequence[int],
    n_qubits: int,
    opcode: str,
) -> bool:
    """Return whether SV-011 supports this exact public runtime request."""

    return bool(
        _pauli_rotation_dispatch_enabled()
        and _pauli_rotation_shape_supported(
            state.shape,
            angles.shape,
            qubits=qubits,
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


def _pauli_rotation_kernel_match(*, device_type: str, dtype: str) -> KernelMatchResult:
    """Match the two-qubit Pauli rotation kernel against its catalog contract."""

    return match_kernel_implementations(
        KernelRequest(
            semantic_id="statevector.apply.pauli_rotation_2q.local",
            device=device_type,
            dtype=dtype,
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
    )


@lru_cache(maxsize=None)
def _require_pauli_rotation_kernel(
    *, device_type: str, dtype: str
) -> KernelImplementation:
    """Return the connected SV-011 implementation or fail closed."""

    return _require_cataloged_kernel(
        _pauli_rotation_kernel_match(device_type=device_type, dtype=dtype),
        implementation_id=_IMPLEMENTATION_ID,
        description="two-qubit Pauli rotation kernel",
    )


def _apply_cataloged_pauli_rotation(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    qubits: Sequence[int],
    opcode: str,
) -> torch.Tensor:
    """Execute SV-011 after exact catalog authorization."""

    _require_pauli_rotation_kernel(
        device_type=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
    )
    from ...kernels.triton.statevector_pauli_rotation import (
        apply_complex64_local_pauli_rotation_2q,
    )

    qubit_tuple = tuple(int(qubit) for qubit in qubits)
    if len(qubit_tuple) != 2:
        raise ValueError("two-qubit Pauli rotations require exactly two qubits")
    qubit_pair = (qubit_tuple[0], qubit_tuple[1])
    half_angles = angles / 2
    rotation = torch.complex(torch.cos(half_angles), torch.sin(half_angles))
    return apply_complex64_local_pauli_rotation_2q(
        state,
        rotation,
        qubits=qubit_pair,
        pauli=_PAULI_BY_OPCODE[opcode],
    )


def _try_apply_cataloged_pauli_rotation(
    state: torch.Tensor,
    angles: torch.Tensor,
    *,
    qubits: Sequence[int],
    n_qubits: int,
    opcode: str,
) -> torch.Tensor | None:
    """Route an evidenced Pauli rotation or preserve the reference path."""

    if not _pauli_rotation_kernel_enabled(
        state,
        angles,
        qubits=qubits,
        n_qubits=n_qubits,
        opcode=opcode,
    ):
        return None
    return _apply_cataloged_pauli_rotation(
        state,
        angles,
        qubits=qubits,
        opcode=opcode,
    )


__all__ = (
    "_apply_cataloged_pauli_rotation",
    "_pauli_rotation_dispatch_enabled",
    "_pauli_rotation_kernel_enabled",
    "_pauli_rotation_kernel_match",
    "_pauli_rotation_shape_supported",
    "_require_pauli_rotation_kernel",
    "_try_apply_cataloged_pauli_rotation",
)
