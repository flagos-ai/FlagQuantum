# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApplyLthBloq and ApplyGateToLthQubit.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers

"""Dispatch controlled components through shared bounded register traversal."""

from collections.abc import Callable
from math import prod
from typing import Literal

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..lookup.unary_iteration import append_nested_bounded_unary_tree
from .multi_control import multi_and_ir


def indexed_component_workspace(
    axis_widths: tuple[int, ...], control_count: int
) -> int:
    """Return parent work, excluding child storage and empty-selection recipes."""
    if any(type(width) is not int or width < 0 for width in axis_widths):
        raise ValueError("Indexed components require positive concrete axis widths")
    if type(control_count) is not int or control_count < 0:
        raise ValueError("Indexed components require a nonnegative control count")
    if not any(axis_widths):
        return 0
    if 0 in axis_widths:
        raise ValueError(
            "Indexed components require positive concrete axis widths for mixed signatures"
        )
    return sum(axis_widths) + control_count - 1


def indexed_component_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    emit_controlled: Callable[[int, int], tuple[Instruction, ...]],
    *,
    controls: tuple[int, ...] = (),
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
    break_early: Callable[[tuple[int, ...], int, int], bool] | None = None,
    emit_zero_selection: Callable[[], tuple[Instruction, ...]] | None = None,
) -> tuple[Instruction, ...]:
    """Apply one child per selected tuple, preserving the source bounded tree.

    Controls are positive. The callback receives the flattened C-order index
    and a positive predicate; it owns child binding and must preserve that
    predicate, the input axes and this parent's workspace. For adjoint traversal,
    it must emit the child adjoint; the parent does not invert child IR.
    Early folding receives the retained parent-axis indices and node interval.
    A folded interval must have an identical controlled child recipe throughout.
    A root fold delegates the entire operation and its direction to the explicit
    zero-selection recipe, which owns any child storage.
    """
    required = indexed_component_workspace(tuple(map(len, axes)), len(controls))
    if type(adjoint) is not bool:
        raise ValueError("Indexed traversal direction must be a boolean")
    if len(axes) != len(extents) or any(
        type(size) is not int or size < 1 or size > 1 << len(axis)
        for axis, size in zip(axes, extents, strict=True)
    ):
        raise ValueError("Every axis needs a concrete extent fitting its word")
    if len(workspace) != required:
        raise ValueError(f"Indexed components require {required} clean work wires")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Indexed components require one distinct classical slot per work wire"
        )
    wires = (*controls, *(wire for axis in axes for wire in axis), *workspace)
    if wires:
        _validate_wires(wires)
    if not any(axes) or (break_early is not None and break_early((), 0, extents[0])):
        if emit_zero_selection is None:
            raise ValueError("Folded first-axis root requires a zero-selection recipe")
        return emit_zero_selection()
    if not controls and extents[0] == 1:
        raise ValueError("Source uncontrolled singleton first axis has no traversal")
    prefix = max(0, len(controls) - 1)
    predicate = (
        controls[0] if len(controls) == 1 else workspace[prefix - 1] if prefix else None
    )
    gates: list[Instruction] = []
    if prefix:
        gates.extend(multi_and_ir(controls, (1,) * len(controls), workspace[:prefix]))

    def visit(index: int, active: int) -> None:
        gates.extend(emit_controlled(index, active))

    append_nested_bounded_unary_tree(
        gates,
        predicate,
        axes,
        extents,
        workspace[prefix:],
        classical_bits[prefix:],
        visit,
        adjoint=adjoint,
        break_early=break_early,
    )
    if prefix:
        gates.extend(
            multi_and_ir(
                controls,
                (1,) * len(controls),
                workspace[:prefix],
                classical_bits=classical_bits[:prefix],
                uncompute=True,
            )
        )
    return tuple(gates)


def indexed_target_component_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    targets: tuple[int, ...],
    emit_controlled: Callable[[int, int, int], tuple[Instruction, ...]],
    *,
    controls: tuple[int, ...] = (),
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Bind a controlled child to its selected target in C-order layout.

    The child receives the flattened index, positive predicate, and selected
    target wire. It owns the emitted child direction and any child storage.
    """
    if len(targets) != prod(extents):
        raise ValueError("Target-position dispatch needs one target per address")
    _validate_wires(
        (*targets, *controls, *(wire for axis in axes for wire in axis), *workspace)
    )
    return indexed_component_ir(
        axes,
        extents,
        lambda index, predicate: emit_controlled(index, predicate, targets[index]),
        controls=controls,
        workspace=workspace,
        classical_bits=classical_bits,
        adjoint=adjoint,
    )


def indexed_pauli_targets_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    targets: tuple[int, ...],
    paulis: tuple[Literal["i", "x", "y", "z"], ...],
    *,
    controls: tuple[int, ...] = (),
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply the selected Pauli to the selected target in C-order layout."""
    if (
        len(targets) != prod(extents)
        or len(paulis) != len(targets)
        or any(pauli not in ("i", "x", "y", "z") for pauli in paulis)
    ):
        raise ValueError(
            "Target-position dispatch needs one Pauli and target per address"
        )

    def emit(index: int, predicate: int, target: int) -> tuple[Instruction, ...]:
        gate = paulis[index]
        if gate == "i":
            return ()
        return (Instruction("c" + gate, (predicate, target)),)

    return indexed_target_component_ir(
        axes,
        extents,
        targets,
        emit,
        controls=controls,
        workspace=workspace,
        classical_bits=classical_bits,
        adjoint=adjoint,
    )
