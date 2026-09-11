"""Shared classical-condition helpers for dynamic circuit layers."""

from ...core.ir import Instruction
from .circuit import DynamicCircuit


def instruction_conditions(
    instruction: Instruction,
) -> tuple[tuple[int, int], ...]:
    if "condition_clauses" in instruction.metadata:
        raise ValueError("complex condition clauses require a DNF-aware execution path")
    if "conditions" in instruction.metadata:
        return tuple(
            (int(bit), int(value)) for bit, value in instruction.metadata["conditions"]
        )
    return ()


def instruction_condition_clauses(
    instruction: Instruction,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if (
        "conditions" in instruction.metadata
        and "condition_clauses" in instruction.metadata
    ):
        raise ValueError(
            "instruction cannot define both conditions and condition_clauses"
        )
    if "condition_clauses" in instruction.metadata:
        try:
            return tuple(
                tuple((int(bit), int(value)) for bit, value in clause)
                for clause in instruction.metadata["condition_clauses"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "condition_clauses must contain integer bit/value pairs"
            ) from exc
    conditions = instruction_conditions(instruction)
    return (conditions,) if conditions else ()


def classical_width(circuit: DynamicCircuit) -> int:
    width = 0
    for instruction in circuit._instructions:
        if instruction.name == "measure":
            width = max(width, int(instruction.metadata["classical_bit"]) + 1)
        for clause in instruction_condition_clauses(instruction):
            for bit, _value in clause:
                width = max(width, bit + 1)
    return width


__all__ = (
    "classical_width",
    "instruction_condition_clauses",
    "instruction_conditions",
)
