"""Dispatch boundary for the native single-qubit CPU adjoint operator."""

from __future__ import annotations

import os
from importlib import import_module
from typing import cast

import torch

_GATE_KINDS = {"rx": 0, "ry": 1, "rz": 2, "rzz": 3}
_EXTENSION_LOADED = False
_EXTENSION_ERROR: Exception | None = None


def _enabled() -> bool:
    return os.getenv("FQ_NATIVE_CPU_ADJOINT", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _load_extension() -> bool:
    global _EXTENSION_ERROR, _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return True
    if _EXTENSION_ERROR is not None:
        return False
    try:
        import_module("flagquantum.simulation.native_cpu._C")
    except (ImportError, OSError) as error:
        _EXTENSION_ERROR = error
        return False
    _EXTENSION_LOADED = True
    return True


def native_cpu_adjoint_available() -> bool:
    """Return whether the installed native operator is enabled and loadable."""

    return _enabled() and _load_extension()


def fused_rotation_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    *,
    name: str,
    wire: int,
    second_wire: int | None = None,
    n_wires: int,
) -> torch.Tensor | None:
    """Mutate ket and adjoint to the pre-gate state and return one VJP.

    ``None`` is the stable fallback signal. The caller retains the existing
    PyTorch implementation for disabled, unavailable, or unsupported cases.
    """

    gate_kind = _GATE_KINDS.get(name)
    if (
        gate_kind is None
        or not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or matrix.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or matrix.dtype != ket.dtype
        or (name == "rzz" and second_wire is None)
        or not ket.is_contiguous()
        or not adjoint.is_contiguous()
        or not matrix.is_contiguous()
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rotation_adjoint_(
                ket,
                adjoint,
                matrix,
                wire,
                -1 if second_wire is None else second_wire,
                n_wires,
                gate_kind,
            ),
        )


def fused_rzz_segment_adjoint_(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    angles: torch.Tensor,
    first_wires: torch.Tensor,
    second_wires: torch.Tensor,
    *,
    n_wires: int,
    aggregate_shared_parameter: bool = False,
) -> torch.Tensor | None:
    """Undo a commuting RZZ segment and return one VJP per gate."""

    real_dtype = torch.float32 if ket.dtype == torch.complex64 else torch.float64
    if (
        not native_cpu_adjoint_available()
        or ket.device.type != "cpu"
        or adjoint.device.type != "cpu"
        or ket.dtype not in {torch.complex64, torch.complex128}
        or adjoint.dtype != ket.dtype
        or angles.device.type != "cpu"
        or angles.dtype != real_dtype
        or first_wires.device.type != "cpu"
        or second_wires.device.type != "cpu"
        or first_wires.dtype != torch.int64
        or second_wires.dtype != torch.int64
        or angles.numel() < 2
        or not all(
            item.is_contiguous()
            for item in (ket, adjoint, angles, first_wires, second_wires)
        )
    ):
        return None
    with torch.no_grad():
        return cast(
            torch.Tensor,
            torch.ops.flagquantum_native.fused_rzz_segment_adjoint_(
                ket,
                adjoint,
                angles,
                first_wires,
                second_wires,
                n_wires,
                aggregate_shared_parameter,
            ),
        )
