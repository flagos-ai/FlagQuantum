# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicallyControlled.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/classically_controlled.py

"""Guard lowered quantum instructions by concrete classical register values."""

from dataclasses import replace

from ...core.classical_control import instruction_condition_clauses
from ...core.ir import Instruction
from .control_predicate import ControlRegister, bind_control_registers


def classically_controlled_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    instructions: tuple[Instruction, ...],
) -> tuple[Instruction, ...]:
    """Apply the child iff all classical groups equal their specified values.

    Bits refer to earlier measurement or classical XOR results. The caller
    retains ownership of those slots. A preexisting child condition is
    conjoined with the wrapper condition. The flat IR admits guarded quantum
    gates and has no guarded measurement, reset, or classical write.
    """
    bits, values = bind_control_registers(controls, registers)
    if not bits:
        return instructions
    fixed = dict(zip(bits, values, strict=True))
    guard = frozenset(fixed.items())
    gates = []
    for instruction in instructions:
        if instruction.name in {"measure", "reset", "classical_xor"}:
            raise ValueError(
                "Classical control cannot guard measurement, reset, or a classical write"
            )
        child_clauses = instruction_condition_clauses(instruction) or ((),)
        possible = {
            guard.union(clause)
            for clause in child_clauses
            if all(bit not in fixed or fixed[bit] == value for bit, value in clause)
        }
        if not possible:
            continue
        clauses = tuple(
            sorted(
                (
                    tuple(sorted(clause))
                    for clause in possible
                    if not any(other < clause for other in possible)
                ),
                key=lambda clause: (len(clause), clause),
            )
        )
        metadata = dict(instruction.metadata)
        metadata.pop("conditions", None)
        metadata.pop("condition_clauses", None)
        if len(clauses) == 1:
            metadata["conditions"] = clauses[0]
        else:
            metadata["condition_clauses"] = clauses
        gates.append(replace(instruction, metadata=metadata))
    return tuple(gates)
