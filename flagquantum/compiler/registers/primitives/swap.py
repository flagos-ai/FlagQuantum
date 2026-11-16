# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitSwap and Swap, and locked Kickmix swap dispatch.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/ops/swap.cc

"""Ordered swaps of physical quantum and classical operands."""

from __future__ import annotations

from ....core.classical_dataflow import ClassicalXor
from ....core.ir import Instruction
from ..._component_trace.symbolic_math import SymbolicInt
from ..bits.bitwise import _validate_wires
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from ._mixed_gates import _Control, _XBasisTarget


def _classical_swap_ir(left: int, right: int) -> tuple[Instruction, ...]:
    return tuple(
        ClassicalXor((target, control), target).instruction()
        for target, control in ((left, right), (right, left), (left, right))
    )


def _broadcast_swap_ir(
    left: tuple[int | ClassicalFieldCoefficient | _XBasisTarget, ...],
    right: tuple[int | ClassicalFieldCoefficient | _XBasisTarget, ...],
) -> tuple[Instruction, ...]:
    """Swap matching physical types lane by lane, retaining classical aliases."""
    if len(left) != len(right):
        raise ValueError("Swap requires equal register widths")
    gates: list[Instruction] = []
    for first, second in zip(left, right, strict=True):
        if type(first) is int and type(second) is int:
            _validate_wires((first, second))
            gates.append(Instruction("swap", (first, second)))
        elif isinstance(first, ClassicalFieldCoefficient) and isinstance(
            second, ClassicalFieldCoefficient
        ):
            gates.extend(_classical_swap_ir(first.slot, second.slot))
        elif (
            isinstance(first, _XBasisTarget)
            and isinstance(second, _XBasisTarget)
            and isinstance(first.outcome, ClassicalFieldCoefficient)
            and isinstance(second.outcome, ClassicalFieldCoefficient)
        ):
            gates.extend(_classical_swap_ir(first.outcome.slot, second.outcome.slot))
        else:
            raise ValueError("Swap lane operands require matching writable types")
    return tuple(gates)


def _broadcast_controlled_swap_ir(
    control: _Control, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Emit ordered physical SWAP/CSWAP with the source control dispatch."""
    if len(left) != len(right):
        raise ValueError("Swap requires equal register widths")
    if control is False:
        return ()
    if control is True:
        return _broadcast_swap_ir(left, right)
    if isinstance(control, ClassicalFieldCoefficient):
        return tuple(
            Instruction(
                gate.name, gate.wires, metadata={"conditions": ((control.slot, 1),)}
            )
            for gate in _broadcast_swap_ir(left, right)
        )
    gates = []
    for first, second in zip(left, right, strict=True):
        _validate_wires((control, first, second))
        gates.append(Instruction("cswap", (control, first, second)))
    return tuple(gates)


def swap_bit_ir(left: int, right: int) -> tuple[Instruction, ...]:
    """Emit the source atomic SWAP on two distinct wires."""
    return swap_register_ir((left,), (right,))


def swap_register_ir(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Exchange corresponding pairs in declared order, with no auxiliary wires."""
    if not left or len(left) != len(right):
        raise ValueError("Swap requires equal nonempty register widths")
    _validate_wires((*left, *right))
    return _broadcast_swap_ir(left, right)


def source_swap_resource_formula(width: SymbolicInt) -> SymbolicResourceFormula:
    """Count one abstract Clifford per pair and retain the original wire width."""
    if isinstance(width, (int, float)) and (type(width) is not int or width < 1):
        raise ValueError("Swap resources require a positive integer width")
    return SymbolicResourceFormula(0, width, 0, 2 * width)
