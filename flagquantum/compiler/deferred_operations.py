"""Lower structured instructions before execution."""

from collections.abc import Sequence
from dataclasses import replace

import torch

from ..core.ir import Instruction
from ..errors import ValidationError


def lower_deferred_operations(
    instructions: Sequence[Instruction],
) -> tuple[Instruction, ...]:
    """Materialize deferred instructions without changing the source program."""
    return tuple(
        (
            _LOWERINGS[instruction.name](instruction)
            if instruction.matrix is None and instruction.name in _LOWERINGS
            else instruction
        )
        for instruction in instructions
    )


def _lower_majorana(instruction: Instruction) -> Instruction:
    if len(instruction.wires) > 10:
        raise ValidationError("The dense SELECT executor supports at most 10 qubits.")

    target_width = instruction.params["target_width"]
    selection_width = instruction.params["selection_width"]
    target_gate = instruction.params["target_gate"]
    dimension = 1 << len(instruction.wires)
    enabled_start = dimension - (1 << (target_width + selection_width))
    matrix = torch.eye(dimension, dtype=torch.complex128)
    for address in range(target_width):
        mask = 1 << (target_width - 1 - address)
        start = enabled_start + (address << target_width)
        for state in range(1 << target_width):
            phase = (-1) ** (state >> (target_width - address)).bit_count()
            bit = bool(state & mask)
            column = start + state
            if target_gate == "Z":
                matrix[column, column] = phase * (-1 if bit else 1)
            else:
                matrix[column, column] = 0
                matrix[column ^ mask, column] = (
                    phase if target_gate == "X" else phase * (-1j if bit else 1j)
                )
    return replace(instruction, matrix=matrix)


def _lower_select_pauli_lcu(instruction: Instruction) -> Instruction:
    if len(instruction.wires) > 10:
        raise ValidationError("The dense SELECT executor supports at most 10 qubits.")

    params = instruction.params
    target_size = 1 << params["target_width"]
    offset = (params["control_val"] or 0) << params["selection_width"]
    matrix = torch.eye(1 << len(instruction.wires), dtype=torch.complex128)
    paulis = {
        "I": torch.eye(2, dtype=matrix.dtype),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=matrix.dtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=matrix.dtype),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=matrix.dtype),
    }
    for address, word in enumerate(params["select_unitaries"]):
        block = torch.tensor([[-1 if word.startswith("-") else 1]], dtype=matrix.dtype)
        for letter in word.removeprefix("-"):
            block = torch.kron(block, paulis[letter])
        start = (offset + address) * target_size
        matrix[start : start + target_size, start : start + target_size] = block
    return replace(instruction, matrix=matrix)


_LOWERINGS = {
    "selected_majorana_fermion": _lower_majorana,
    "select_pauli_lcu": _lower_select_pauli_lcu,
}
