# The early-stop rule is adapted from Qualtran v0.7.0.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Measurement-assisted traversal of a quantum address register."""

from collections.abc import Callable

from ...core.ir import Instruction


def unary_iteration_workspace(selection_width: int, control_value: int | None) -> int:
    """Count the clean workspace qubits reserved by the traversal."""
    return max(selection_width + int(control_value is not None) - 1, 0)


def append_unary_iteration(
    instructions: list[Instruction],
    *,
    operand_count: int,
    width: int,
    entries: int,
    control_value: int | None,
    apply_branch: Callable[[int, int | None], None],
    coherent: bool = False,
    break_early: Callable[[int, int], bool] | None = None,
    classical_bits: tuple[int, ...] = (),
) -> None:
    """Visit populated addresses in ascending order and clear all workspace.

    Operands start with the optional control and the big-endian address.
    Clean workspace follows the operands. The callback receives the address
    and its active-high predicate qubit, or None for an unconditional leaf.
    It may use that predicate as a control or for a phase, and must preserve
    the computational values of the predicate and address.
    """
    control_width = int(control_value is not None)

    def emit(name: str, *qubits: int) -> None:
        instructions.append(Instruction(name, qubits))

    def visit(control: int, depth: int, start: int, size: int) -> None:
        if start >= entries:
            return
        if size == 1 or (
            break_early is not None
            and start + size <= entries
            and break_early(start, start + size)
        ):
            apply_branch(start, control)
            return
        right = control_width + depth
        work = operand_count + depth - (1 - control_width)
        half = size // 2
        # Compute a clean temporary AND of the prefix and the inverted bit.
        emit("x", right)
        if coherent:
            emit("ccx", control, right, work)
            emit("x", right)
            visit(work, depth + 1, start, half)
            emit("cx", control, work)
            visit(work, depth + 1, start + half, half)
            emit("ccx", control, right, work)
            return
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
        bit = classical_bits[work - operand_count] if classical_bits else work
        instructions.append(
            Instruction(
                "measure", (work,), metadata={"is_dynamic": True, "classical_bit": bit}
            )
        )
        instructions.append(
            Instruction("cz", (control, right), metadata={"conditions": ((bit, 1),)})
        )
        instructions.append(
            Instruction("reset", (work,), metadata={"is_dynamic": True})
        )

    if control_value is not None:
        if control_value == 0:
            emit("x", 0)
        visit(0, 0, 0, 1 << width)
        if control_value == 0:
            emit("x", 0)
    elif width:
        half = 1 << (width - 1)
        emit("x", 0)
        visit(0, 1, 0, half)
        emit("x", 0)
        visit(0, 1, half, half)
    else:
        apply_branch(0, None)
