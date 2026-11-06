"""Lower structured Pauli SELECT to the canonical circuit representation."""

from dataclasses import replace
from typing import Literal

from ..core.classical_control import instruction_condition_clauses
from ..core.ir import CircuitIR, Instruction
from ..errors import ValidationError
from .lowering.deferred_operations import (
    lower_deferred_operations,
    lower_majorana_to_gates,
    lower_select_pauli_lcu_to_gates,
)
from .registers.unary_iteration import append_unary_iteration, unary_iteration_workspace

SelectPath = Literal["dense", "dynamic_unary_iteration", "unitary_ladder"]
SELECT_REPORT = "select_lowering"


def is_structured_select(instruction: Instruction) -> bool:
    return instruction.name == "select_pauli_lcu" and instruction.matrix is None


def _is_structured_majorana(instruction: Instruction) -> bool:
    return (
        instruction.name == "selected_majorana_fermion" and instruction.matrix is None
    )


def supports_dynamic_select(instruction: Instruction) -> bool:
    """Return whether the instruction can use the measurement-assisted traversal."""
    return is_structured_select(instruction) or (
        _is_structured_majorana(instruction)
        and instruction.params["control_width"] <= 1
    )


def unary_select_ir(instruction: Instruction) -> CircuitIR:
    """Build the measurement-assisted decomposition in operand order."""
    if not supports_dynamic_select(instruction):
        raise ValidationError("Expected a structured SELECT with at most one control.")
    params = instruction.params
    width = params["selection_width"]
    controlled = params["control_width"]
    majorana = _is_structured_majorana(instruction)
    operands = len(instruction.wires)
    targets = range(controlled + width, operands)
    entries = len(targets) if majorana else len(params["select_unitaries"])
    control_value = (1 if majorana else params["control_val"]) if controlled else None
    ladder_width = unary_iteration_workspace(width, control_value)
    # A full address space guarantees exactly one accumulator toggle on
    # each enabled branch. Partial tables use explicit Jordan–Wigner strings.
    use_accumulator = majorana and width > 0 and entries == 1 << width
    accumulator = operands + ladder_width
    instructions: list[Instruction] = []

    def emit(name: str, *qubits: int) -> None:
        instructions.append(Instruction(name, qubits))

    def apply_word(address: int, control: int | None) -> None:
        if majorana:
            pauli = params["target_gate"].lower()
            if use_accumulator:
                assert control is not None
                emit("cx", control, accumulator)
                emit("c" + pauli, control, targets[address])
                emit("cz", accumulator, targets[address])
                return
            word = "Z" * address + pauli.upper() + "I" * (entries - address - 1)
        else:
            word = params["select_unitaries"][address]
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

    if use_accumulator:
        if controlled:
            emit("cx", 0, accumulator)
        else:
            emit("x", accumulator)

    append_unary_iteration(
        instructions,
        operand_count=operands,
        width=width,
        entries=entries,
        control_value=control_value,
        apply_branch=apply_word,
    )
    return CircuitIR(accumulator + int(use_accumulator), tuple(instructions))


def lower_select_program(ir: CircuitIR, path: SelectPath) -> CircuitIR:
    """Expand SELECT operations and record the path and reusable workspace."""

    def should_lower(instruction: Instruction) -> bool:
        return is_structured_select(instruction) or (
            path != "dense" and _is_structured_majorana(instruction)
        )

    if not any(map(should_lower, ir.instructions)):
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
        if not should_lower(op):
            instructions.append(op)
            continue
        chosen_path = path
        if path == "dense":
            instructions.extend(lower_deferred_operations((op,)))
            needed = 0
        else:
            if path == "dynamic_unary_iteration" and supports_dynamic_select(op):
                fragment = unary_select_ir(op)
            elif _is_structured_majorana(op):
                chosen_path = "unitary_ladder"
                fragment = lower_majorana_to_gates(op).to_ir()
            else:
                fragment = lower_select_pauli_lcu_to_gates(op).to_ir()
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
            {
                "instruction_index": index,
                "path": chosen_path,
                "workspace_qubits": needed,
            }
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
