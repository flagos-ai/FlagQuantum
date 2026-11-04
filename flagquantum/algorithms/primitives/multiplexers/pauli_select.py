"""Measurement-assisted unary iteration for structured Pauli SELECT."""

from __future__ import annotations

from ....compiler.select_lowering import unary_select_ir
from ....core.ir import Instruction
from ....runtime.dynamic.circuit import DynamicCircuit


def lower_select_pauli_lcu_to_unary_iteration(
    instruction: Instruction,
) -> DynamicCircuit:
    """Build an executable dynamic circuit for one structured SELECT."""
    return DynamicCircuit.from_ir(unary_select_ir(instruction))
