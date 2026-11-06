# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Bounded traversal adapted from Qualtran v0.7.0 unary_iteration_bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Measurement-assisted traversal of a quantum address register."""

from collections.abc import Callable

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir


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
        if size == 1:
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


def append_bounded_unary_tree(
    gates: list[Instruction],
    control: int | None,
    selection: tuple[int, ...],
    extent: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    visit_leaf: Callable[[int, int], None],
    *,
    break_early: Callable[[int, int], bool] | None = None,
    adjoint: bool = False,
) -> None:
    """Visit the source's bounded tree or reverse its component sequence."""
    compute_values = (1, 1) if adjoint else (1, 0)
    erase_values = (1, 0) if adjoint else (1, 1)

    def visit(predicate: int, depth: int, start: int, stop: int) -> None:
        if stop <= extent and (
            stop - start == 1 or (break_early is not None and break_early(start, stop))
        ):
            visit_leaf(start, predicate)
            return
        middle = (start + stop) // 2
        if extent <= middle:
            visit(predicate, depth + 1, start, middle)
            return
        wire, work = selection[depth], workspace[depth]
        halves = ((start, middle), (middle, stop))
        first, second = reversed(halves) if adjoint else halves
        gates.extend(and_compute_ir(predicate, wire, work, values=compute_values))
        visit(work, depth + 1, *first)
        gates.append(Instruction("cx", (predicate, work)))
        visit(work, depth + 1, *second)
        gates.extend(
            and_uncompute_ir(
                predicate, wire, work, classical_bits[depth], values=erase_values
            )
        )

    capacity = 1 << len(selection)
    if control is not None:
        visit(control, 0, 0, capacity)
        return
    # The first address bit that distinguishes populated halves is the predicate.
    while extent <= capacity // 2:
        selection = selection[1:]
        capacity //= 2
    predicate, selection = selection[0], selection[1:]
    middle = capacity // 2
    flip = Instruction("x", (predicate,))
    if adjoint:
        visit(predicate, 0, middle, capacity)
        gates.append(flip)
        visit(predicate, 0, 0, middle)
        gates.append(flip)
    else:
        gates.append(flip)
        visit(predicate, 0, 0, middle)
        gates.append(flip)
        visit(predicate, 0, middle, capacity)


def append_nested_bounded_unary_tree(
    gates: list[Instruction],
    control: int | None,
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    visit_leaf: Callable[[int, int], None],
) -> None:
    """Visit axes of extent at least two and flatten only the output index."""

    def walk(depth: int, prefix: int, predicate: int | None, used: int) -> None:
        word, extent = axes[depth], extents[depth]
        stop = used + unary_iteration_workspace(len(word), predicate)

        def leaf(index: int, active: int) -> None:
            flat_index = prefix * extent + index
            if depth + 1 == len(axes):
                visit_leaf(flat_index, active)
            else:
                walk(depth + 1, flat_index, active, stop)

        append_bounded_unary_tree(
            gates,
            predicate,
            word,
            extent,
            workspace[used:stop],
            classical_bits[used:stop],
            leaf,
        )

    walk(0, 0, control, 0)
