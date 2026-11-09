# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Resource model adapted from the Qualtran v0.7.0 grid Hamiltonian QVR.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/grid_ham/qvr.py

"""Grid phase estimation costs and exact register phase application."""

from math import isfinite, ldexp

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...core.ir import Instruction


def source_grid_qvr_cost(width: int) -> ComponentCost:
    """Preserve one call to an unspecified-angle Rz per phase bit."""
    if type(width) is not int or width < 1:
        raise ValueError("Grid QVR requires a positive phase width")
    rotation = ComponentCost("Rz", 1, GateCost(rotation=1))
    return ComponentCost(
        "QuantumVariableRotation",
        width,
        calls=(CostCall(rotation, width, width - 1),),
    )


def grid_qvr_ir(
    phase: tuple[int, ...], *, radians_per_unit: float
) -> tuple[Instruction, ...]:
    """Apply exp(i * radians_per_unit * word) to an unsigned big-endian word."""
    _validate_wires(phase)
    if not isfinite(radians_per_unit):
        raise ValueError("Grid QVR requires a finite phase coefficient")
    return tuple(
        Instruction("phase", (wire,), {"theta": ldexp(radians_per_unit, bit)})
        for bit, wire in enumerate(reversed(phase))
    )
