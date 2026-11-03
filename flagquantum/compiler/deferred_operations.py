"""Lower structured instructions before execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..core.ir import Instruction
from ..errors import ValidationError

if TYPE_CHECKING:
    from ..circuit import Circuit


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


def lower_select_pauli_lcu_to_gates(instruction: Instruction) -> Circuit:
    """Decompose a structured SELECT into a circuit in operand order.

    The first qubits correspond to `instruction.wires`. Any remaining qubits
    are ladder workspace and must start in zero. The circuit restores them to
    zero. Use `Circuit.compose` to bind operands and workspace to a larger
    circuit.
    """
    from ..algorithms.primitives.oracle import append_multi_controlled_x
    from ..circuit import Circuit

    if instruction.name != "select_pauli_lcu" or instruction.matrix is not None:
        raise ValidationError("Expected a structured select_pauli_lcu instruction.")

    params = instruction.params
    operand_count = len(instruction.wires)
    control_count = params["control_width"] + params["selection_width"]
    controls = tuple(range(control_count))
    targets = range(control_count, operand_count)
    scratch = tuple(range(operand_count, operand_count + max(control_count - 2, 0)))
    circuit = Circuit(operand_count + len(scratch))
    basis_change = {"Y": ("sdg", "s"), "Z": ("h", "h")}

    for address, word in enumerate(params["select_unitaries"]):
        pattern = ((params["control_val"] or 0) << params["selection_width"]) | address
        flips = [q for q in controls if not (pattern >> (control_count - q - 1)) & 1]
        for qubit in flips:
            circuit.x(qubit)

        if word.startswith("-"):
            if control_count > 1:
                circuit.h(controls[-1])
                append_multi_controlled_x(
                    circuit, controls[:-1], controls[-1], ancillas=scratch[:-1]
                )
                circuit.h(controls[-1])
            elif control_count == 1:
                circuit.z(controls[0])
            else:
                # XZXZ is -I, including its phase relative to a caller's reference.
                target = targets[0]
                circuit.x(target).z(target).x(target).z(target)

        for target, pauli in zip(targets, word.removeprefix("-"), strict=True):
            if pauli == "I":
                continue
            if not controls:
                circuit.gate(pauli.lower(), target)
                continue
            change = basis_change.get(pauli)
            if change is not None:
                circuit.gate(change[0], target)
            append_multi_controlled_x(circuit, controls, target, ancillas=scratch)
            if change is not None:
                circuit.gate(change[1], target)

        for qubit in flips:
            circuit.x(qubit)

    return circuit


_LOWERINGS = {
    "selected_majorana_fermion": _lower_majorana,
    "select_pauli_lcu": _lower_select_pauli_lcu,
}
