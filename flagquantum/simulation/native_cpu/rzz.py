"""Dispatch boundary for fused native CPU RZZ segments."""

from __future__ import annotations

import os
from typing import cast

import torch

from .adjoint import _load_extension


def _enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_RZZ_FUSION", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def native_cpu_rzz_available() -> bool:
    """Return whether fused package-local CPU RZZ execution is available."""

    return _enabled() and _load_extension()


def fused_rzz_segment_forward_(
    state: torch.Tensor,
    angles: torch.Tensor,
    first_wires: torch.Tensor,
    second_wires: torch.Tensor,
    *,
    n_wires: int,
) -> bool:
    """Apply a contiguous RZZ segment in one state traversal when supported."""

    real_dtype = torch.float32 if state.dtype == torch.complex64 else torch.float64
    if (
        not native_cpu_rzz_available()
        or state.device.type != "cpu"
        or state.dtype not in {torch.complex64, torch.complex128}
        or angles.device.type != "cpu"
        or angles.dtype != real_dtype
        or first_wires.device.type != "cpu"
        or second_wires.device.type != "cpu"
        or first_wires.dtype != torch.int64
        or second_wires.dtype != torch.int64
        or state.ndim != 2
        or angles.ndim != 1
        or angles.numel() < 2
        or first_wires.shape != angles.shape
        or second_wires.shape != angles.shape
        or (torch.is_grad_enabled() and angles.requires_grad)
        or not all(
            item.is_contiguous() for item in (state, angles, first_wires, second_wires)
        )
    ):
        return False
    with torch.no_grad():
        cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rzz_segment_forward_(
                state, angles, first_wires, second_wires, n_wires
            ),
        )
    return True
