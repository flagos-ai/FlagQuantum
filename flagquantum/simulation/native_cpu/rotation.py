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
    wires: torch.Tensor,
    *,
    n_wires: int,
) -> bool:
    """Apply two to six disjoint one-qubit matrices to a state in place.

    ``False`` is the stable fallback signal. The caller retains the existing
    PyTorch path when the extension is disabled, unavailable, or unsupported.
    """

    if (
        not native_cpu_rotation_available()
        or state.device.type != "cpu"
        or matrices.device.type != "cpu"
        or wires.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or matrices.dtype != state.dtype
        or wires.dtype != torch.int64
        or state.ndim != 2
        or matrices.ndim != 3
        or matrices.shape[0] not in {2, 3, 4, 5, 6}
        or matrices.shape[1:] != (2, 2)
        or wires.shape != (matrices.shape[0],)
        or (torch.is_grad_enabled() and matrices.requires_grad)
        or not all(item.is_contiguous() for item in (state, matrices, wires))
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rotation_block_forward_(
                state, matrices, wires, n_wires
            ),
        )
    return True


def fused_rotation_block_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrices: torch.Tensor,
    wires: torch.Tensor,
    *,
    n_wires: int,
) -> bool:
    """Apply one legacy fixed block to ket and adjoint separately."""

    if not fused_rotation_block_forward_(ket, matrices, wires, n_wires=n_wires):
        return False
    if not fused_rotation_block_forward_(adjoint, matrices, wires, n_wires=n_wires):
        raise RuntimeError("native fixed block rejected a matching adjoint state")
    return True


def fused_hadamard_block_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    wires: torch.Tensor,
    *,
    n_wires: int,
) -> bool:
    """Apply a disjoint Hadamard block to ket and adjoint in place."""

    if (
        not native_cpu_hadamard_block_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or wires.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or wires.dtype != torch.int64
        or ket.ndim != 2
        or adjoint.shape != ket.shape
        or not 2 <= wires.numel() <= 11
        or wires.ndim != 1
        or not all(item.is_contiguous() for item in (ket, adjoint, wires))
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_hadamard_block_adjoint_(
                ket, adjoint, wires, n_wires
            ),
        )
    return True
