"""Lower structured Pauli SELECT to the canonical circuit representation."""

from dataclasses import replace
from typing import Literal

from ..core.classical_control import instruction_condition_clauses
from ..core.ir import CircuitIR, Instruction
from ..errors import ValidationError
from .deferred_operations import (
    lower_deferred_operations,
    lower_select_pauli_lcu_to_gates,
)

SelectPath = Literal["dense", "dynamic_unary_iteration", "unitary_ladder"]
SELECT_REPORT = "select_lowering"


def is_structured_select(instruction: Instruction) -> bool:
    return instruction.name == "select_pauli_lcu" and instruction.matrix is None


def unary_select_ir(instruction: Instruction) -> CircuitIR:
    """Build the measurement-assisted decomposition in operand order."""
    if not is_structured_select(instruction):
        raise ValidationError("Expected a structured select_pauli_lcu instruction.")
    params = instruction.params
    width = params["selection_width"]
    controlled = params["control_width"]
    words = params["select_unitaries"]
    operands = len(instruction.wires)
    targets = range(controlled + width, operands)
    instructions: list[Instruction] = []

    def emit(name: str, *qubits: int) -> None:
        instructions.append(Instruction(name, qubits))

    def apply_word(address: int, control: int | None) -> None:
        word = words[address]
        if word.startswith("-"):
            if control is None:
                for name in ("x", "z", "x", "z"):
                    emit(name, targets[0])
            else:
                emit("z", control)
        for target, pauli in zip(targets, word.removeprefix("-"), strict=True):
            if pauli != "I":
                if control is None:
                    emit(pauli.lower(), target)
                else:
                    emit("c" + pauli.lower(), control, target)

    def visit(control: int, depth: int, start: int, size: int) -> None:
        if start >= len(words):
            return
        if size == 1:
            apply_word(start, control)
            return
        right = controlled + depth
        work = operands + depth - (1 - controlled)
        half = size // 2
        # Compute a clean temporary AND of the prefix and the inverted bit.
        emit("x", right)
        emit("h", work)
        emit("t", work)
        for parent in (control, right):
            emit("cx", parent, work)
        for parent in (control, right):
            emit("cx", work, parent)
        for parent in (control, right):
            emit("tdg", parent)
        emit("t", work)
        for parent in (control, right):
            emit("cx", work, parent)
        emit("h", work)
        emit("s", work)
        emit("x", right)
        visit(work, depth + 1, start, half)
        emit("cx", control, work)
        visit(work, depth + 1, start + half, half)
        emit("h", work)
        instructions.append(
            Instruction(
                "measure", (work,), metadata={"is_dynamic": True, "classical_bit": work}
            )
        )
        instructions.append(
            Instruction("cz", (control, right), metadata={"conditions": ((work, 1),)})
        )
        instructions.append(
            Instruction("reset", (work,), metadata={"is_dynamic": True})
        )

    if controlled:
        if params["control_val"] == 0:
            emit("x", 0)
        visit(0, 0, 0, 1 << width)
        if params["control_val"] == 0:
            emit("x", 0)
    elif width:
        half = 1 << (width - 1)
        emit("x", 0)
        visit(0, 1, 0, half)
        emit("x", 0)
        visit(0, 1, half, half)
    else:
        apply_word(0, None)
    return CircuitIR(operands + max(width + controlled - 1, 0), tuple(instructions))


def lower_select_program(ir: CircuitIR, path: SelectPath) -> CircuitIR:
    """Expand SELECT operations and record the path and reusable workspace."""
    if not any(map(is_structured_select, ir.instructions)):
        return ir
    instructions: list[Instruction] = []
    workspace = 0
    records = []
    # Keep compiler feedback bits separate from all user classical bits.
    next_bit = 1 + max(
        (
            bit
            for op in ir.instructions
            for bit in (
                op.metadata.get("classical_bit", -1),
                *(
                    bit
                    for clause in instruction_condition_clauses(op)
                    for bit, _ in clause
                ),
            )
        ),
        default=-1,
    )
    for index, op in enumerate(ir.instructions):
        if not is_structured_select(op):
            instructions.append(op)
            continue
        if path == "dense":
            instructions.extend(lower_deferred_operations((op,)))
            needed = 0
        else:
            fragment = (
                unary_select_ir(op)
                if path == "dynamic_unary_iteration"
                else lower_select_pauli_lcu_to_gates(op).to_ir()
            )
            needed = fragment.n_wires - len(op.wires)
            mapping = (*op.wires, *range(ir.n_wires, ir.n_wires + needed))
            for item in fragment.instructions:
                metadata = dict(item.metadata)
                if item.name == "measure":
                    metadata["classical_bit"] = (
                        next_bit + metadata["classical_bit"] - len(op.wires)
                    )
                if "conditions" in metadata:
                    metadata["conditions"] = tuple(
                        (next_bit + bit - len(op.wires), value)
                        for bit, value in metadata["conditions"]
                    )
                instructions.append(
                    replace(
                        item,
                        wires=tuple(mapping[q] for q in item.wires),
                        metadata=metadata,
                    )
                )
        workspace = max(workspace, needed)
        records.append(
            {"instruction_index": index, "path": path, "workspace_qubits": needed}
        )
    width = ir.n_wires + workspace
    metadata = {
        **ir.metadata,
        SELECT_REPORT: {
            "logical_qubits": ir.n_wires,
            "workspace_qubits": workspace,
            "operations": records,
        },
    }
    shape = ir.shape if not workspace else (*ir.shape[:-1], 2**width)
    return replace(
        ir,
        instructions=tuple(instructions),
        n_wires=width,
        shape=shape,
        metadata=metadata,
    )
