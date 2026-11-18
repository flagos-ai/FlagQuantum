# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Equals and EqualsAConstant.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, explicit work and shared multi-control reduction.

"""Register and constant equality through the same multi-controlled X primitive."""

from ....core.ir import Instruction
from ..._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..bits.bitwise import xor_ir
from ..control.multi_control import (
    multi_control_x_ir,
    source_multi_control_x_cost,
    source_multi_control_x_resource_formula,
)
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def equality_constant_ir(
    source: tuple[int, ...],
    value: int,
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR x==value into target, leaving source and all clean work unchanged."""
    width = len(source)
    if width < 1 or type(value) is not int or not 0 <= value < 1 << width:
        raise ValueError("Equality constant must fit a nonempty unsigned register")
    pattern = tuple((value >> (width - i - 1)) & 1 for i in range(width))
    return multi_control_x_ir(source, pattern, target, workspace, classical_bits)


def equality_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR a==b into target, following Xor -> all-zero MultiControlX -> Xor.

    Registers have equal positive width and big-endian bit order. Supply n-1
    initially zero work qubits and n-1 measurement slots (none for one bit).
    Inputs, target phases and clean work are preserved except for the target XOR.
    """
    if not left or len(left) != len(right):
        raise ValueError("Equality requires equal positive input widths")
    wires = (*left, *right, target, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Equality requires distinct nonnegative quantum wires")
    middle = multi_control_x_ir(
        right, (0,) * len(right), target, workspace, classical_bits
    )
    xor = xor_ir(left, right)
    return (*xor, *middle, *xor)


def source_equality_cost(width: int) -> ComponentCost:
    """Retain two word XORs and the original all-zero controlled X model."""
    if type(width) is not int or width < 1:
        raise ValueError("Equality model requires a positive integer width")
    return ComponentCost(
        "Equals",
        2 * width + 1,
        GateCost(clifford=2 * width),
        (CostCall(source_multi_control_x_cost((0,) * width), bystander_qubits=width),),
    )


def source_equality_constant_cost(width: int, value: int) -> ComponentCost:
    """Retain the original binary constant pattern and controlled X model."""
    if (
        type(width) is not int
        or width < 1
        or type(value) is not int
        or not 0 <= value < 1 << width
    ):
        raise ValueError("Equality constant model requires a fitting unsigned value")
    pattern = tuple((value >> i) & 1 for i in reversed(range(width)))
    return ComponentCost(
        "EqualsAConstant",
        width + 1,
        calls=(CostCall(source_multi_control_x_cost(pattern)),),
    )


def source_equality_resource_formula(width: SymbolicInt) -> SymbolicResourceFormula:
    """Preserve all-zero concrete controls versus unspecified symbolic controls."""
    if type(width) is int:
        concrete = source_equality_cost(width)
        report = estimate_component_resources(concrete)
        return SymbolicResourceFormula(
            report.gates.and_bloq,
            report.gates.clifford,
            report.gates.measurement,
            report.peak_logical_qubits,
        )
    control = source_multi_control_x_resource_formula(width)
    return SymbolicResourceFormula(
        control.and_bloq,
        control.clifford + 2 * width,
        control.measurement,
        smax(2, 2 * width + 1, control.logical_qubits),
    )


def source_equality_constant_resource_formula(
    width: SymbolicInt, value: SymbolicInt
) -> SymbolicResourceFormula:
    """Use the actual constant pattern only when both fields are Python ints."""
    if type(width) is int and type(value) is int:
        concrete = source_equality_constant_cost(width, value)
        report = estimate_component_resources(concrete)
        return SymbolicResourceFormula(
            report.gates.and_bloq,
            report.gates.clifford,
            report.gates.measurement,
            report.peak_logical_qubits,
        )
    control = source_multi_control_x_resource_formula(width)
    return SymbolicResourceFormula(
        control.and_bloq,
        control.clifford,
        control.measurement,
        smax(width + 1, control.logical_qubits),
    )
