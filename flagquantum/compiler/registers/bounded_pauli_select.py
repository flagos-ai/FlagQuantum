# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectPauliLCU and its unary iteration costs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/select_pauli_lcu.py

"""Bind signed Pauli leaves to the source bounded traversal."""

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost, GateCost
from .bitwise import _validate_wires
from .unary_iteration import append_bounded_unary_tree, unary_iteration_workspace


def _selection_parameters(
    instruction: Instruction,
) -> tuple[tuple[str, ...], int, int | None]:
    if instruction.name != "select_pauli_lcu" or instruction.matrix is not None:
        raise ValueError("Bounded traversal requires a structured Pauli SELECT")
    words: tuple[str, ...] = instruction.params["select_unitaries"]
    width: int = instruction.params["selection_width"]
    value: int | None = instruction.params["control_val"]
    if width == 0 or (value is None and len(words) < 2):
        raise ValueError(
            "Source bounded Pauli SELECT requires an address and two rows without a control"
        )
    return words, width, value


def bounded_pauli_select_ir(
    instruction: Instruction,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Emit the source stream, including its behavior on padded addresses."""
    words, width, value = _selection_parameters(instruction)
    work = unary_iteration_workspace(width, value)
    if len(workspace) != work:
        raise ValueError(f"Bounded Pauli SELECT requires {work} clean work wires")
    if (
        len(classical_bits) != work
        or len(set(classical_bits)) != work
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Bounded Pauli SELECT requires distinct classical slots for its work"
        )
    _validate_wires((*instruction.wires, *workspace))
    offset = int(value is not None)
    control = instruction.wires[0] if offset else None
    selection = instruction.wires[offset : offset + width]
    targets = instruction.wires[offset + width :]
    gates: list[Instruction] = []

    def visit(index: int, predicate: int) -> None:
        word = words[index]
        for pauli, target in zip(word.removeprefix("-"), targets, strict=True):
            if pauli != "I":
                gates.append(Instruction("c" + pauli.lower(), (predicate, target)))
        if word.startswith("-"):
            gates.append(Instruction("z", (predicate,)))

    flip = () if control is None or value == 1 else (Instruction("x", (control,)),)
    gates.extend(flip)
    append_bounded_unary_tree(
        gates, control, selection, len(words), workspace, classical_bits, visit
    )
    gates.extend(flip)
    return tuple(gates)


def source_bounded_pauli_select_cost(instruction: Instruction) -> ComponentCost:
    """Count locked logical gates and the peak active width."""
    words, _, value = _selection_parameters(instruction)
    control_width = int(value is not None)
    ands = len(words) - 2 + control_width
    active_work = (len(words) - 1).bit_length() + control_width - 1
    # Each nonidentity letter is a controlled Pauli, and a minus sign adds Z.
    word_cliffords = sum(len(word) - word.count("I") for word in words)
    cliffords = 2 * ands + 2 * int(value in (None, 0)) + word_cliffords
    return ComponentCost(
        "SelectPauliLCU",
        len(instruction.wires) + active_work,
        GateCost(and_bloq=ands, clifford=cliffords, measurement=ands),
    )
