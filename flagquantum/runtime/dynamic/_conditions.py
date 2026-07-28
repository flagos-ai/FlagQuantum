"""Shared classical-condition helpers for dynamic circuit layers."""

from ...core.ir import Instruction
from .circuit import DynamicCircuit


def instruction_conditions(
    instruction: Instruction,
) -> tuple[tuple[int, int], ...]:
    if "conditions" in instruction.metadata:
        return tuple(
            (int(bit), int(value)) for bit, value in instruction.metadata["conditions"]
        )
    legacy = instruction.metadata.get("condition")
    if legacy:
        return ((int(legacy["bit"]), int(legacy["equals"])),)
    return ()


def classical_width(circuit: DynamicCircuit) -> int:
    width = 0
    for instruction in circuit._instructions:
        if instruction.name == "measure":
            width = max(width, int(instruction.metadata["classical_bit"]) + 1)
        for bit, _value in instruction_conditions(instruction):
            width = max(width, bit + 1)
    return width


__all__ = ("classical_width", "instruction_conditions")
