"""Dispatch boundary for fused native CPU rotation blocks."""

from __future__ import annotations

import os
from typing import cast

import torch

from .adjoint import _load_extension


def _enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_ROTATION_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def native_cpu_rotation_available() -> bool:
    """Return whether fused package-local CPU rotations are enabled and loadable."""

    return _enabled() and _load_extension()


def native_cpu_one_qubit_layer_available() -> bool:
    """Return whether generic disjoint one-qubit layer fusion is enabled."""

    return (
        os.getenv("FQ_NATIVE_CPU_ONE_QUBIT_LAYER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and native_cpu_rotation_available()
    )


def native_cpu_static_clifford_layer_available() -> bool:
    """Return whether exact static Clifford layer fusion is enabled."""

    return (
        os.getenv("FQ_CPU_NATIVE_STATIC_CLIFFORD_LAYER", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and native_cpu_one_qubit_layer_available()
    )


def fused_static_clifford_layer_(
    state: torch.Tensor,
    gate_codes: torch.Tensor,
    qubits: torch.Tensor,
    *,
    n_qubits: int,
) -> bool:
    """Apply one disjoint H/S/Sdg/X/Y/Z layer in place."""

    if (
        not native_cpu_static_clifford_layer_available()
        or state.device.type != "cpu"
        or gate_codes.device.type != "cpu"
        or qubits.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or gate_codes.dtype != torch.int8
        or qubits.dtype != torch.int64
        or state.ndim != 2
        or gate_codes.ndim != 1
        or qubits.shape != gate_codes.shape
        or not 2 <= qubits.numel() <= 62
        or not all(item.is_contiguous() for item in (state, gate_codes, qubits))
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_static_clifford_layer_(
                state, gate_codes, qubits, n_qubits
            ),
        )
    return True


def native_cpu_shared_rzz_forward_fusion_available() -> bool:
    """Return whether a shared RZZ segment may fuse into a rotation block."""

    return os.getenv(
        "FQ_NATIVE_CPU_SHARED_RZZ_FORWARD_FUSION", "1"
    ).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def native_cpu_specialized_forward_rotations_available() -> bool:
    """Return whether forward tiles may use specialized rotation arithmetic."""

    return os.getenv(
        "FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def native_cpu_forward_rotation_tile_qubits(n_qubits: int) -> int:
    """Return the bounded forward tile width with an explicit rollback."""

    if n_qubits < 16:
        return 4
    enabled = os.getenv(
        "FQ_NATIVE_CPU_FORWARD_WIDE_ROTATION_TILES", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}
    return 8 if enabled else 6


def native_cpu_hadamard_block_adjoint_available() -> bool:
    """Return whether paired wide Hadamard blocks are enabled and loadable."""

    return (
        os.getenv("FQ_NATIVE_CPU_ADJOINT_WIDE_TILES", "1").strip().lower()
        not in {"0", "false", "off", "no"}
        and native_cpu_one_qubit_layer_available()
    )


def fused_rotation_block_forward_(
    state: torch.Tensor,
    matrices: torch.Tensor,
    qubits: torch.Tensor,
    *,
    n_qubits: int,
    rzz_angles: torch.Tensor | None = None,
    rzz_first_qubits: torch.Tensor | None = None,
    rzz_second_qubits: torch.Tensor | None = None,
) -> bool:
    """Apply two to eleven shared or batch-specific one-qubit matrices in place.

    ``False`` is the stable fallback signal. The caller retains the existing
    PyTorch path when the extension is disabled, unavailable, or unsupported.
    """

    rzz_tensors = (rzz_angles, rzz_first_qubits, rzz_second_qubits)
    real_dtype = torch.float32 if state.dtype == torch.complex64 else torch.float64
    if (
        not native_cpu_rotation_available()
        or state.device.type != "cpu"
        or matrices.device.type != "cpu"
        or qubits.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or matrices.dtype != state.dtype
        or qubits.dtype != torch.int64
        or state.ndim != 2
        or matrices.ndim not in {3, 4}
        or (matrices.ndim == 4 and matrices.shape[0] != state.shape[0])
        or not 2 <= matrices.shape[-3] <= 11
        or matrices.shape[-2:] != (2, 2)
        or qubits.shape != (matrices.shape[-3],)
        or (torch.is_grad_enabled() and matrices.requires_grad)
        or not all(item.is_contiguous() for item in (state, matrices, qubits))
        or (
            any(item is None for item in rzz_tensors)
            and any(item is not None for item in rzz_tensors)
        )
        or (
            rzz_angles is not None
            and (
                not native_cpu_shared_rzz_forward_fusion_available()
                or rzz_angles.device.type != "cpu"
                or rzz_angles.dtype != real_dtype
                or rzz_first_qubits is None
                or rzz_second_qubits is None
                or rzz_first_qubits.device.type != "cpu"
                or rzz_second_qubits.device.type != "cpu"
                or rzz_first_qubits.dtype != torch.int64
                or rzz_second_qubits.dtype != torch.int64
                or rzz_angles.ndim != 1
                or rzz_angles.numel() < 2
                or rzz_first_qubits.shape != rzz_angles.shape
                or rzz_second_qubits.shape != rzz_angles.shape
                or not torch.equal(rzz_angles, rzz_angles[0].expand_as(rzz_angles))
                or not bool(
                    torch.all(torch.abs(rzz_first_qubits - rzz_second_qubits) == 1)
                )
                or torch.unique(
                    torch.minimum(rzz_first_qubits, rzz_second_qubits)
                ).numel()
                != rzz_angles.numel()
                or not rzz_angles.is_contiguous()
                or not rzz_first_qubits.is_contiguous()
                or not rzz_second_qubits.is_contiguous()
            )
        )
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rotation_block_forward_(
                state,
                matrices,
                qubits,
                n_qubits,
                rzz_angles,
                rzz_first_qubits,
                rzz_second_qubits,
                native_cpu_specialized_forward_rotations_available(),
            ),
        )
    return True


def fused_rotation_block_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrices: torch.Tensor,
    qubits: torch.Tensor,
    *,
    n_qubits: int,
) -> bool:
    """Apply one legacy fixed block to ket and adjoint separately."""

    if not fused_rotation_block_forward_(ket, matrices, qubits, n_qubits=n_qubits):
        return False
    if not fused_rotation_block_forward_(adjoint, matrices, qubits, n_qubits=n_qubits):
        raise RuntimeError("native fixed block rejected a matching adjoint state")
    return True


def fused_hadamard_block_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    qubits: torch.Tensor,
    *,
    n_qubits: int,
) -> bool:
    """Apply a disjoint Hadamard block to ket and adjoint in place."""

    if (
        not native_cpu_hadamard_block_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or qubits.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or qubits.dtype != torch.int64
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or not 2 <= qubits.numel() <= 11
        or qubits.ndim != 1
        or not all(item.is_contiguous() for item in (ket, adjoint, qubits))
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_hadamard_block_adjoint_(
                ket, adjoint, qubits, n_qubits
            ),
        )
    return True
